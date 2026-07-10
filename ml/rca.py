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
    "supplier_manufacturing_data.csv"
)

sensor_df = pd.read_csv(sensor_file)

supplier_df = pd.read_csv(supplier_file)

merged = sensor_df.merge(
    supplier_df,
    on="supplier_id",
    how="left"
)

def calculate_rca(row):

    score = 0

    if row["anomaly_flag"] == -1:
        score += 30

    if row["ppm"] > 30000:
        score += 30

    if row["audit_score"] < 70:
        score += 20

    if row["repeat_failure_rate"] > 0.20:
        score += 20

    if score >= 80:
        return "Supplier Manufacturing Defect"

    elif score >= 50:
        return "Process Variation"

    return "Normal"

merged["root_cause"] = merged.apply(
    calculate_rca,
    axis=1
)

output_file = os.path.join(
    BASE_DIR,
    "supplier_sensor_rca.csv"
)

merged.to_csv(
    output_file,
    index=False
)

print("\nRCA Summary\n")
print(
    merged["root_cause"].value_counts()
)

print(
    "\nSaved:",
    output_file
)
