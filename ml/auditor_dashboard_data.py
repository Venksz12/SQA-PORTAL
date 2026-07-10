import pandas as pd

sensor = pd.read_csv("sensor_anomaly_results.csv")
supplier = pd.read_csv("supplier_predictions.csv")
rca = pd.read_csv("unified_rca_results.csv")

print("\n===== AUDITOR KPI =====\n")

print(
    "Total Sensor Records:",
    len(sensor)
)

print(
    "Sensor Anomalies:",
    (sensor["anomaly_flag"] == -1).sum()
)

print(
    "Risk Suppliers:",
    len(
        supplier[
            supplier[
                "predicted_supplier_score"
            ] < 0.30
        ]
    )
)

print(
    "Critical RCA Cases:",
    len(
        rca[
            rca["rca_score"] >= 80
        ]
    )
)