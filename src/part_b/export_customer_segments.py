
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CLUSTER_FILE = ROOT / "outputs" / "part_b" / "customer_clusters.csv"
PROFILE_FILE = ROOT / "outputs" / "part_b" / "customer_cluster_profiles.csv"
OUTPUT_FILE = ROOT / "outputs" / "customer_segments.csv"

SEGMENT_NAMES = {
    0: "Occasional Customers",
    1: "High-Value Premium Customers",
    2: "Loyal Frequent Customers",
    3: "At-Risk Dormant Customers"
}


def main():
    print("Generating customer segments...")

    clusters = pd.read_csv(CLUSTER_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    required_cluster_columns = {
        "customer_id", "recency", "frequency",
        "monetary", "cluster_id"
    }

    if not required_cluster_columns.issubset(clusters.columns):
        raise ValueError("Unexpected cluster CSV columns.")

    if len(clusters) != 1175:
        raise ValueError(f"Expected 1175 customers, found {len(clusters)}")

    if clusters["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found.")

    # Identify Premium dynamically using highest mean monetary value.
    premium_cluster = int(
        profiles.loc[
            profiles["avg_monetary"].idxmax(),
            "cluster_id"
        ]
    )

    clusters["segment_name"] = (
        clusters["cluster_id"].map(SEGMENT_NAMES)
    )

    if clusters["segment_name"].isna().any():
        raise ValueError("A cluster does not have a segment name.")

    clusters["is_premium"] = (
        clusters["cluster_id"] == premium_cluster
    )

    result = clusters[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "cluster_id",
            "segment_name",
            "is_premium"
        ]
    ].copy()

    result = result.rename(
        columns={"cluster_id": "segment_id"}
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n========== SEGMENT SUMMARY ==========")
    print(result.groupby(
        ["segment_id", "segment_name", "is_premium"]
    ).size().to_string())

    print(f"\nPremium cluster: {premium_cluster}")
    print(f"Premium customers: {result['is_premium'].sum()}")
    print(f"Total customers: {len(result)}")
    print(f"\nSaved: {OUTPUT_FILE}")
    print("\nCUSTOMER SEGMENT EXPORT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
