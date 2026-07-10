import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "sensor_stream_data.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

df = pd.read_csv(CSV_PATH)

model = joblib.load(
    os.path.join(MODEL_DIR,"sensor_iforest.joblib")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR,"sensor_scaler.joblib")
)

features = joblib.load(
    os.path.join(MODEL_DIR,"sensor_features.joblib")
)

X = df[features].fillna(0)

X_scaled = scaler.transform(X)

df["anomaly_flag"] = model.predict(X_scaled)

print(df["anomaly_flag"].value_counts())

anomalies = df[
    df["anomaly_flag"] == -1
]

print("\nTop Anomalies:\n")
print(anomalies.head())

output_file = os.path.join(
    BASE_DIR,
    "sensor_anomaly_results.csv"
)

df.to_csv(output_file, index=False)

print("\nSaved:", output_file)