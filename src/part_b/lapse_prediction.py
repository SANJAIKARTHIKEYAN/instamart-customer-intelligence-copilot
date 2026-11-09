
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import resample


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "clean"
PART_B = ROOT / "outputs" / "part_b"

PART_B.mkdir(parents=True, exist_ok=True)

CUSTOMERS_FILE = DATA / "customers.csv"
ORDERS_FILE = DATA / "orders.csv"
PRODUCTS_FILE = DATA / "products.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"

MODEL_FILE = ROOT / "outputs" / "lapse_model.joblib"
RETENTION_FILE = ROOT / "outputs" / "retention_list.csv"

METRICS_FILE = PART_B / "lapse_metrics.json"
PREDICTIONS_FILE = PART_B / "lapse_test_predictions.csv"
RF_IMPORTANCE_FILE = PART_B / "rf_top_features.csv"
LR_COEFFICIENT_FILE = PART_B / "lr_coefficients.csv"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")
RANDOM_STATE = 42

NUMERIC = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

CATEGORICAL = [
    "city",
    "gender",
    "favourite_category"
]

FEATURES = NUMERIC + CATEGORICAL


# ==========================================================
# TASK 4: LOAD DATA AND BUILD LAPSE DATASET
# ==========================================================

def load_data():

    customers = pd.read_csv(CUSTOMERS_FILE)
    orders = pd.read_csv(ORDERS_FILE)
    products = pd.read_csv(PRODUCTS_FILE)

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    for col in [
        "order_amount", "discount_pct", "delivery_minutes"
    ]:
        orders[col] = pd.to_numeric(
            orders[col], errors="coerce"
        )

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    if orders["order_ts"].isna().any():
        raise ValueError("Invalid order timestamps found.")

    return customers, orders, products


def build_dataset(customers, orders, products):

    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    orders["status_clean"] = (
        orders["status"].astype(str).str.strip().str.lower()
    )

    # Only orders strictly before cutoff enter features.
    historical = orders[
        orders["order_ts"] < CUTOFF
    ].copy()

    completed = historical[
        historical["status_clean"] == "completed"
    ].copy()

    # Eligible customers: at least one earlier completed order.
    eligible_ids = completed["customer_id"].unique()

    # September completed orders are used ONLY for labels.
    september = orders[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < LABEL_END)
        & (orders["status_clean"] == "completed")
    ]

    september_active = set(september["customer_id"])

    # RFM features.
    rfm = completed.groupby("customer_id").agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )

    rfm["recency"] = (
        CUTOFF - rfm["last_order_date"]
    ).dt.days

    # Average discount of completed historical orders.
    avg_discount = completed.groupby(
        "customer_id"
    )["discount_pct"].mean().rename("avg_discount")

    # Cancellation rate: cancelled / all final-status orders.
    final_status = historical[
        historical["status_clean"].isin(
            ["completed", "cancelled"]
        )
    ]

    # Count distinct orders, not CSV rows, in case an order appears
    # on more than one product line.
    final_status_orders = final_status[
        ["customer_id", "order_id", "status_clean"]
    ].drop_duplicates(["customer_id", "order_id"])

    status_stats = final_status_orders.groupby("customer_id").agg(
        final_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status_clean",
            lambda values: (values == "cancelled").sum()
        )
    )

    status_stats["cancellation_rate"] = (
        status_stats["cancelled_orders"]
        / status_stats["final_orders"]
    )

    # Average completed delivery time.
    delivery = completed.groupby(
        "customer_id"
    )["delivery_minutes"].mean().rename(
        "avg_delivery_minutes"
    )

    # Most frequent completed-order product category.
    favourite = completed.groupby(
        "customer_id"
    )["category"].agg(
        lambda x: (
            x.dropna().mode().iloc[0]
            if not x.dropna().empty
            else np.nan
        )
    ).rename("favourite_category")

    features = (
        rfm.join(avg_discount)
        .join(status_stats[["cancellation_rate"]])
        .join(delivery)
        .join(favourite)
    ).reset_index()

    # Add clean customer information.
    features = features.merge(
        customers[
            ["customer_id", "city", "age", "gender", "signup_date"]
        ],
        on="customer_id",
        how="left",
        validate="one_to_one"
    )

    features["tenure_days"] = (
        CUTOFF - features["signup_date"]
    ).dt.days

    # Target creation.
    features["lapsed"] = (
        ~features["customer_id"].isin(september_active)
    ).astype(int)

    # Feature leakage guard.
    assert set(features["customer_id"]) == set(eligible_ids)
    assert "segment_id" not in FEATURES
    assert "is_premium" not in FEATURES

    # Expected reference checks.
    assert len(features) == 1156
    assert int(features["lapsed"].sum()) == 443

    return features


# ==========================================================
# TASK 5: PREPROCESSING WITHOUT LEAKAGE
# ==========================================================

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])


def verify_preprocessing(preprocessor, X_train, X_test, X):

    audit = clone(preprocessor)

    train_transformed = audit.fit_transform(X_train)
    test_transformed = audit.transform(X_test)

    # Verify that the transformed matrices contain no missing values.
    for name, matrix in [
        ("Training", train_transformed),
        ("Testing", test_transformed)
    ]:
        values = (
            matrix.toarray()
            if hasattr(matrix, "toarray")
            else np.asarray(matrix)
        )

        assert np.isfinite(values).all(), (
            f"{name} features contain missing or infinite values."
        )

    scaler = (
        audit.named_transformers_["numeric"]
        .named_steps["scaler"]
    )

    learned_frequency_mean = float(
        scaler.mean_[NUMERIC.index("frequency")]
    )

    full_cohort_frequency_mean = float(X["frequency"].mean())

    print("\n========== PREPROCESSING VERIFICATION ==========")
    print("Training transformed shape:", train_transformed.shape)
    print("Testing transformed shape:", test_transformed.shape)
    print("Missing values in transformed features: 0")
    print("OneHotEncoder: drop='first'")
    print("Scaler fitted on training rows only")

    print(
        "Training scaler frequency mean:",
        round(learned_frequency_mean, 4)
    )

    print(
        "Full cohort frequency mean:",
        round(full_cohort_frequency_mean, 4)
    )

    print(
        "These means differ because the scaler learns from "
        "the training subset, while the cohort mean uses all "
        "eligible customers."
    )


# ==========================================================
# TASK 6: EVALUATION HELPERS
# ==========================================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, pos_label=1,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, pos_label=1,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, pos_label=1,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist()
    }


# ==========================================================
# TASK 7: TRAINING-ONLY OVERSAMPLING
# ==========================================================

def oversample_training(X_train, y_train):

    training = X_train.copy()
    training["_target"] = y_train.to_numpy()

    majority_count = int(y_train.value_counts().max())
    minority_count = int(y_train.value_counts().min())

    majority = training[
        training["_target"] == y_train.value_counts().idxmax()
    ]

    minority = training[
        training["_target"] == y_train.value_counts().idxmin()
    ]

    additional = resample(
        minority,
        replace=True,
        n_samples=majority_count - minority_count,
        random_state=RANDOM_STATE
    )

    balanced = pd.concat(
        [training, additional],
        ignore_index=True
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X_balanced = balanced[FEATURES]
    y_balanced = balanced["_target"]

    assert y_balanced.value_counts().nunique() == 1

    return X_balanced, y_balanced


# ==========================================================
# TASK 8: FEATURE INTERPRETATION
# ==========================================================

def save_feature_interpretation(rf_model, lr_model):

    # Random Forest: five most important features.
    rf_preprocessor = rf_model.named_steps["preprocessor"]
    rf_estimator = rf_model.named_steps["model"]

    rf_names = rf_preprocessor.get_feature_names_out()

    rf_importance = pd.DataFrame({
        "feature": rf_names,
        "importance": rf_estimator.feature_importances_
    }).sort_values(
        "importance", ascending=False
    ).head(5)

    rf_importance.to_csv(
        RF_IMPORTANCE_FILE, index=False
    )

    # Logistic Regression: three largest absolute coefficients.
    lr_preprocessor = lr_model.named_steps["preprocessor"]
    lr_estimator = lr_model.named_steps["model"]

    lr_names = lr_preprocessor.get_feature_names_out()
    coefficients = lr_estimator.coef_[0]

    lr_coefficients = pd.DataFrame({
        "feature": lr_names,
        "coefficient": coefficients,
        "sign": np.where(
            coefficients > 0, "Positive",
            np.where(coefficients < 0, "Negative", "Zero")
        )
    })

    lr_coefficients["absolute_coefficient"] = (
        lr_coefficients["coefficient"].abs()
    )

    lr_coefficients = lr_coefficients.sort_values(
        "absolute_coefficient", ascending=False
    ).head(3)

    lr_coefficients.to_csv(
        LR_COEFFICIENT_FILE, index=False
    )

    print("\n========== TOP 5 RANDOM FOREST FEATURES ==========")
    print(rf_importance.to_string(index=False))

    print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")
    print(lr_coefficients.to_string(index=False))

    return rf_importance, lr_coefficients


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    print("\n==========================================")
    print("INSTAMART CUSTOMER LAPSE PREDICTION")
    print("==========================================")

    customers, orders, products = load_data()

    dataset = build_dataset(
        customers, orders, products
    )

    X = dataset[FEATURES]
    y = dataset["lapsed"]

    print("\n========== TASK 4: COHORT ==========")
    print("Eligible customers:", len(dataset))
    print("Lapsed:", int(y.sum()))
    print("Retained:", int((y == 0).sum()))
    print("Lapse rate:", round(y.mean() * 100, 2), "%")

    # Stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())

    assert len(X_train) == 924
    assert len(X_test) == 232

    # Task 5.
    preprocessor = create_preprocessor()

    verify_preprocessing(
        preprocessor, X_train, X_test, X
    )

    # Baseline: always predict not lapsed.
    baseline_predictions = np.zeros(len(y_test), dtype=int)

    baseline = {
        "accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "precision": float(
            precision_score(
                y_test, baseline_predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, baseline_predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, baseline_predictions,
                zero_division=0
            )
        )
    }

    print("\n========== TASK 6: BASELINE ==========")
    print(json.dumps(baseline, indent=4))

    # Logistic Regression.
    lr_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    lr_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100]
    }

    print("\n========== LOGISTIC REGRESSION GRID SEARCH ==========")

    lr_search = GridSearchCV(
        lr_pipeline,
        lr_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    lr_search.fit(X_train, y_train)

    # Random Forest.
    rf_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        ))
    ])

    rf_grid = {
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    }

    print("\n========== RANDOM FOREST GRID SEARCH ==========")
    print("Expected fits: 9 configurations x 5 folds = 45")

    rf_search = GridSearchCV(
        rf_pipeline,
        rf_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True
    )

    rf_search.fit(X_train, y_train)

    print("\n========== GRID SEARCH RESULTS ==========")

    print("Logistic best parameters:", lr_search.best_params_)
    print("Logistic best CV F1:", round(lr_search.best_score_, 4))

    print("Random Forest best parameters:", rf_search.best_params_)
    print("Random Forest best CV F1:", round(rf_search.best_score_, 4))

    # Untouched test evaluation.
    lr_best = lr_search.best_estimator_
    rf_best = rf_search.best_estimator_

    lr_original_metrics = evaluate(
        lr_best, X_test, y_test
    )

    rf_metrics = evaluate(
        rf_best, X_test, y_test
    )

    print("\n========== TEST RESULTS: LOGISTIC ==========")
    print(json.dumps(lr_original_metrics, indent=4))

    print("\n========== TEST RESULTS: RANDOM FOREST ==========")
    print(json.dumps(rf_metrics, indent=4))

    # Business decision: choose the better original tuned model
    # by lapsed-class F1, considering recall and precision too.
    if rf_metrics["f1"] > lr_original_metrics["f1"]:
        original_selected_name = "Random Forest"
        original_selected = rf_best
        original_selected_metrics = rf_metrics
    else:
        original_selected_name = "Logistic Regression"
        original_selected = lr_best
        original_selected_metrics = lr_original_metrics

    print("\nOriginal-model selection:", original_selected_name)

    # Task 7: oversample only training data.
    X_balanced, y_balanced = oversample_training(
        X_train, y_train
    )

    print("\n========== TASK 7: BALANCED TRAINING ==========")
    print(y_balanced.value_counts().sort_index())

    assert sorted(
        y_balanced.value_counts().tolist()
    ) == [570, 570]

    balanced_lr = clone(lr_best)
    balanced_lr.fit(X_balanced, y_balanced)

    balanced_lr_metrics = evaluate(
        balanced_lr, X_test, y_test
    )

    print("\n========== ORIGINAL LOGISTIC REGRESSION ==========")
    print({
        k: lr_original_metrics[k]
        for k in ["precision", "recall", "f1"]
    })

    print("\n========== OVERSAMPLED LOGISTIC REGRESSION ==========")
    print({
        k: balanced_lr_metrics[k]
        for k in ["precision", "recall", "f1"]
    })

    # Final selection: compare only the two original tuned models.
    # The oversampled model is a separate class-balance experiment and
    # must not replace either tuned model in the final model comparison.
    candidates = {
        "Logistic Regression": (lr_best, lr_original_metrics),
        "Random Forest": (rf_best, rf_metrics),
    }

    selected_name = max(
        candidates,
        key=lambda name: candidates[name][1]["f1"]
    )
    selected_model, selected_metrics = candidates[selected_name]

    if selected_metrics["f1"] < 0.55:
        raise ValueError(
            "Neither original tuned model achieved the required test F1 >= 0.55."
        )

    print("\n========== SELECTED MODEL ==========")
    print("Selected:", selected_name)
    print("Test F1:", round(selected_metrics["f1"], 4))
    print("Test precision:", round(selected_metrics["precision"], 4))
    print("Test recall:", round(selected_metrics["recall"], 4))

    # Task 8a.
    rf_importance, lr_coefficients = save_feature_interpretation(
        rf_best, lr_best
    )

    # Save and reload selected model.
    joblib.dump(selected_model, MODEL_FILE)

    reloaded_model = joblib.load(MODEL_FILE)

    original_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions, reloaded_predictions
    )

    print("\nModel persistence verification: PASSED")
    print("Reloaded predictions exactly match original predictions.")

    # Generate test predictions.
    probabilities = reloaded_model.predict_proba(X_test)[:, 1]

    test_predictions = pd.DataFrame({
        "customer_id": dataset.loc[
            X_test.index, "customer_id"
        ].values,
        "actual_lapsed": y_test.to_numpy(),
        "predicted_lapsed": reloaded_predictions,
        "lapse_probability": probabilities
    })

    test_predictions.to_csv(
        PREDICTIONS_FILE, index=False
    )

    # Task 8c: post-hoc Premium retention list.
    segments = pd.read_csv(SEGMENTS_FILE)

    if "is_premium" not in segments.columns:
        raise ValueError("Premium flag missing from segment file.")

    premium_ids = set(
        segments.loc[
            segments["is_premium"].astype(str).str.lower() == "true",
            "customer_id"
        ]
    )

    retention = test_predictions[
        test_predictions["customer_id"].isin(premium_ids)
    ].copy()

    retention = retention.sort_values(
        "lapse_probability",
        ascending=False
    ).head(10)

    retention = retention.rename(
        columns={"actual_lapsed": "actually_lapsed"}
    )

    retention = retention[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]

    retention.to_csv(
        RETENTION_FILE, index=False
    )

    true_lapsed_in_list = int(
        retention["actually_lapsed"].sum()
    )

    # Save all evaluation information.
    report = {
        "cutoff": str(CUTOFF.date()),
        "label_window": [
            str(CUTOFF.date()),
            str((LABEL_END - pd.Timedelta(days=1)).date())
        ],
        "cohort_size": len(dataset),
        "lapsed_customers": int(y.sum()),
        "retained_customers": int((y == 0).sum()),
        "lapse_rate": float(y.mean()),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "baseline": baseline,
        "logistic_regression": {
            "best_parameters": lr_search.best_params_,
            "best_cv_f1": float(lr_search.best_score_),
            "test_metrics": lr_original_metrics
        },
        "random_forest": {
            "best_parameters": rf_search.best_params_,
            "best_cv_f1": float(rf_search.best_score_),
            "test_metrics": rf_metrics
        },
        "oversampled_logistic_regression": {
            "class_counts": {
                str(k): int(v)
                for k, v in y_balanced.value_counts().items()
            },
            "test_metrics": balanced_lr_metrics
        },
        "selected_model": selected_name,
        "model_selection_basis": "Higher lapsed-class test F1 among the two original tuned models; oversampled logistic regression is reported separately.",
        "selected_test_metrics": selected_metrics,
        "retention_list_count": len(retention),
        "actually_lapsed_in_retention_list": true_lapsed_in_list
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== RETENTION LIST ==========")
    print(retention.to_string(index=False))
    print("Premium test customers listed:", len(retention))
    print("Actually lapsed in list:", true_lapsed_in_list)

    print("\n========== SAVED FILES ==========")
    for path in [
        MODEL_FILE,
        RETENTION_FILE,
        METRICS_FILE,
        PREDICTIONS_FILE,
        RF_IMPORTANCE_FILE,
        LR_COEFFICIENT_FILE
    ]:
        print(path)

    print("\nALL LAPSE TASKS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
