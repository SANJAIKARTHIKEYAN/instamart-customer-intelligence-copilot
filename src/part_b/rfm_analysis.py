
from pathlib import Path
import pandas as pd

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

ORDERS_FILE = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_b"
OUTPUT_FILE = OUTPUT_DIR / "rfm_analysis.csv"

# Official snapshot moment
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


# ==========================================
# RFM ANALYSIS
# ==========================================

def calculate_rfm():

    print("Starting RFM Customer Analysis...")

    # 1. Load cleaned orders
    orders = pd.read_csv(ORDERS_FILE)

    # 2. Validate required columns
    required_columns = [
        "order_id",
        "customer_id",
        "order_ts",
        "status",
        "order_amount"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in orders.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 3. Convert data types
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"],
        errors="coerce"
    )

    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"],
        errors="coerce"
    )

    if orders["order_ts"].isna().any():
        raise ValueError("Invalid order timestamps found.")

    if orders["order_amount"].isna().any():
        raise ValueError("Invalid order amounts found.")

    # 4. Select completed orders only
    completed_orders = orders[
        orders["status"].str.strip().str.lower() == "completed"
    ].copy()

    if completed_orders.empty:
        raise ValueError("No completed orders found.")

    # 5. Group orders by customer
    rfm = completed_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    # 6. Calculate Recency
    # IMPORTANT: Use the complete timestamp.
    # Do not normalize or remove the time component.
    rfm["recency"] = (
        SNAPSHOT_DATE - rfm["last_order_ts"]
    ).dt.days

    # 7. Arrange final columns
    rfm = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary"
        ]
    ]

    rfm = rfm.sort_values(
        "customer_id"
    ).reset_index(drop=True)

    # 8. Data validation
    assert len(completed_orders) == 11110, (
        f"Expected 11110 completed orders, "
        f"found {len(completed_orders)}"
    )

    assert len(rfm) == 1175, (
        f"Expected 1175 customers, found {len(rfm)}"
    )

    assert rfm["customer_id"].is_unique
    assert rfm["recency"].min() >= 0
    assert (rfm["frequency"] > 0).all()
    assert (rfm["monetary"] > 0).all()
    assert rfm.notna().all().all()

    # 9. Save RFM results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rfm.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.2f"
    )

    # 10. Display results
    print("\n========== RFM ANALYSIS ==========")

    print(f"Snapshot Moment    : {SNAPSHOT_DATE}")
    print(f"Completed Orders   : {len(completed_orders)}")
    print(f"Unique Customers   : {len(rfm)}")

    print("\n========== RFM SUMMARY ==========")

    print(
        rfm[
            ["recency", "frequency", "monetary"]
        ].describe().round(2)
    )

    print("\n========== RFM REFERENCE CHECKS ==========")

    print(f"Recency Median     : {rfm['recency'].median()}")
    print(f"Recency Minimum    : {rfm['recency'].min()}")
    print(f"Recency Maximum    : {rfm['recency'].max()}")

    print(f"Frequency Mean     : {rfm['frequency'].mean():.2f}")
    print(f"Frequency Maximum  : {rfm['frequency'].max()}")

    print(f"Monetary Mean      : {rfm['monetary'].mean():.2f}")
    print(f"Monetary Maximum   : {rfm['monetary'].max():.2f}")

    print("\n========== SAMPLE RECORDS ==========")

    print(rfm.head(10).to_string(index=False))

    print("\n========== OUTPUT ==========")
    print(f"Saved successfully: {OUTPUT_FILE}")

    print("\nALL RFM VALIDATIONS PASSED")
    print("RFM ANALYSIS COMPLETED SUCCESSFULLY")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    calculate_rfm()
