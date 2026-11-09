
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "outputs" / "part_b" / "customer_clusters.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_b"

PCA_FILE = OUTPUT_DIR / "pca_coordinates.csv"
PROFILE_FILE = OUTPUT_DIR / "customer_cluster_profiles.csv"
PLOT_FILE = OUTPUT_DIR / "pca_customer_segments.png"
VARIANCE_FILE = OUTPUT_DIR / "pca_explained_variance.csv"

FEATURES = ["recency", "frequency", "monetary"]


def run_pca_analysis():

    print("Starting PCA and Customer Profiling...")

    # 1. Load clustered customer data
    df = pd.read_csv(INPUT_FILE)

    required = FEATURES + ["customer_id", "cluster_id"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) != 1175:
        raise ValueError(f"Expected 1175 customers, found {len(df)}")

    # 2. Prepare and scale features
    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply PCA: 3 dimensions to 2
    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    # 4. Save PCA coordinates
    pca_df = pd.DataFrame({
        "customer_id": df["customer_id"],
        "cluster_id": df["cluster_id"],
        "PC1": components[:, 0],
        "PC2": components[:, 1]
    })

    # 5. Explained variance
    variance_df = pd.DataFrame({
        "principal_component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent":
            pca.explained_variance_ratio_ * 100
    })

    total_variance = pca.explained_variance_ratio_.sum() * 100

    # 6. Customer profiling
    profiles = df.groupby("cluster_id").agg(
        customer_count=("customer_id", "nunique"),
        avg_recency=("recency", "mean"),
        median_recency=("recency", "median"),
        avg_frequency=("frequency", "mean"),
        median_frequency=("frequency", "median"),
        avg_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median")
    ).reset_index()

    # 7. Generate PCA visualization
    plt.figure(figsize=(10, 7))

    for cluster in sorted(df["cluster_id"].unique()):

        mask = pca_df["cluster_id"] == cluster

        plt.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            label=f"Cluster {cluster}",
            alpha=0.65,
            s=35
        )

    plt.title("Instamart Customer Segmentation - PCA")
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    # 8. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pca_df.to_csv(PCA_FILE, index=False)
    profiles.to_csv(PROFILE_FILE, index=False, float_format="%.2f")
    variance_df.to_csv(VARIANCE_FILE, index=False)

    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    # 9. Display results
    print("\n========== PCA RESULTS ==========")

    print(f"Customers analyzed: {len(pca_df)}")
    print("Original dimensions: 3")
    print("Reduced dimensions: 2")

    print(f"PC1 variance: {pca.explained_variance_ratio_[0] * 100:.2f}%")
    print(f"PC2 variance: {pca.explained_variance_ratio_[1] * 100:.2f}%")
    print(f"Total explained variance: {total_variance:.2f}%")

    print("\n========== CUSTOMER CLUSTER PROFILES ==========")
    print(profiles.round(2).to_string(index=False))

    print("\n========== SAVED FILES ==========")
    print(PCA_FILE)
    print(PROFILE_FILE)
    print(VARIANCE_FILE)
    print(PLOT_FILE)

    assert len(pca_df) == 1175
    assert pca_df[["PC1", "PC2"]].notna().all().all()
    assert profiles["customer_count"].sum() == 1175

    print("\nPCA AND CUSTOMER PROFILING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_pca_analysis()
