import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sensor_file = os.path.join(
    BASE_DIR,
    "sensor_anomaly_results.csv"
)

supplier_file = os.path.join(
    BASE_DIR,
    "supplier_predictions.csv"
)

claim_file = os.path.join(
    BASE_DIR,
    "claim_fraud_predictions.csv"
)

print("Loading datasets...")

sensor_df = pd.read_csv(sensor_file)
supplier_df = pd.read_csv(supplier_file)
claim_df = pd.read_csv(claim_file)

# ==========================
# SENSOR AGGREGATION
# ==========================

sensor_summary = (
    sensor_df
    .groupby("supplier_id")
    .agg(
        anomaly_count=("anomaly_flag",
                       lambda x: (x == -1).sum())
    )
    .reset_index()
)

# ==========================
# CLAIM AGGREGATION
# ==========================

claim_summary = (
    claim_df
    .groupby("supplier_id")
    .agg(
        avg_fraud_probability=(
            "fraud_probability",
            "mean"
        ),

        max_fraud_probability=(
            "fraud_probability",
            "max"
        )
    )
    .reset_index()
)

# ==========================
# SUPPLIER DATA
# ==========================

supplier_summary = supplier_df[
    [
        "supplier_id",
        "ppm",
        "audit_score",
        "repeat_failure_rate",
        "predicted_supplier_score"
    ]
].copy()

# ==========================
# MERGE
# ==========================

merged = supplier_summary.merge(
    sensor_summary,
    on="supplier_id",
    how="left"
)

merged = merged.merge(
    claim_summary,
    on="supplier_id",
    how="left"
)

merged.fillna(0, inplace=True)

# ==========================
# RCA
# ==========================

def rca(row):

    score = 0
    causes = []

    if row["anomaly_count"] > 20:
        score += 30
        causes.append("High Sensor Anomalies")

    if row["ppm"] > 30000:
        score += 25
        causes.append("High PPM")

    if row["audit_score"] < 70:
        score += 20
        causes.append("Low Audit Score")

    if row["repeat_failure_rate"] > 0.20:
        score += 15
        causes.append("Repeat Failures")

    if row["avg_fraud_probability"] > 80:
        score += 30
        causes.append("Fraud Risk")

    if score >= 80:
        category = "Supplier Manufacturing Defect"

    elif score >= 60:
        category = "Critical Process Variation"

    elif score >= 40:
        category = "Potential Quality Issue"

    else:
        category = "Normal"

    return pd.Series([
        score,
        category,
        ", ".join(causes)
    ])

merged[
    [
        "rca_score",
        "root_cause",
        "cause_details"
    ]
] = merged.apply(
    rca,
    axis=1
)

output_file = os.path.join(
    BASE_DIR,
    "unified_rca_results.csv"
)

merged.to_csv(
    output_file,
    index=False
)

print("\nRCA Summary\n")
print(
    merged["root_cause"]
    .value_counts()
)

print("\nTop Risk Suppliers\n")
print(
    merged[
        [
            "supplier_id",
            "rca_score",
            "root_cause"
        ]
    ]
    .sort_values(
        "rca_score",
        ascending=False
    )
    .head(20)
)

print(
    "\nSaved:",
    output_file
)