
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "outputs" / "part_b" / "rfm_analysis.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_b"

EVALUATION_FILE = OUTPUT_DIR / "kmeans_evaluation.csv"
CLUSTER_FILE = OUTPUT_DIR / "customer_clusters.csv"

FEATURES = ["recency", "frequency", "monetary"]


def run_segmentation():

    print("Starting K-Means Customer Segmentation...")

    # 1. Load RFM data
    rfm = pd.read_csv(INPUT_FILE)

    if len(rfm) != 1175:
        raise ValueError(
            f"Expected 1175 customers, found {len(rfm)}"
        )

    # 2. Prepare features
    X = rfm[FEATURES].copy()

    if X.isna().any().any():
        raise ValueError("Missing values found in RFM features.")

    # 3. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Evaluate K = 2 to 8
    evaluation_results = []

    print("\n========== MODEL EVALUATION ==========")

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        evaluation_results.append({
            "k": k,
            "inertia": inertia,
            "silhouette_score": silhouette
        })

        print(
            f"K={k} | "
            f"Inertia={inertia:.2f} | "
            f"Silhouette={silhouette:.3f}"
        )

    # 5. Save evaluation results
    evaluation_df = pd.DataFrame(evaluation_results)

    # 6. Select initial candidate K=4
    selected_k = 4

    final_model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10
    )

    rfm["cluster_id"] = final_model.fit_predict(X_scaled)

    # 7. Save cluster assignments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluation_df.to_csv(
        EVALUATION_FILE,
        index=False
    )

    rfm.to_csv(
        CLUSTER_FILE,
        index=False
    )

    # 8. Display cluster distribution
    print("\n========== CLUSTER DISTRIBUTION ==========")

    print(
        rfm["cluster_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n========== EVALUATION SUMMARY ==========")

    print(evaluation_df.to_string(index=False))

    best_k = int(
        evaluation_df.loc[
            evaluation_df["silhouette_score"].idxmax(),
            "k"
        ]
    )

    print(f"\nHighest silhouette score: K={best_k}")
    print(f"Initial candidate K: {selected_k}")

    print(f"\nEvaluation saved: {EVALUATION_FILE}")
    print(f"Cluster assignments saved: {CLUSTER_FILE}")

    print("\nK-MEANS SEGMENTATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_segmentation()
