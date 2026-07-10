import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Sensor Dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df)}")

numeric_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

exclude_cols = [
    "sensor_id",
    "claim_id",
    "failure_flag",
    "anomaly_label",
    "repair_confirmed"
]

preferred_features = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "engine_load",
    "coolant_temp",
    "oil_pressure",
    "ambient_temp",
    "speed",
    "maintenance_age_days"
]

features = [
    c for c in preferred_features
    if c in df.columns
]

print("\nFeatures Used:")
print(features)

X = df[features].fillna(0)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

print("\nTraining Model...")

model.fit(X_scaled)

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "sensor_iforest.joblib"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "sensor_scaler.joblib"
    )
)

joblib.dump(
    features,
    os.path.join(
        MODEL_DIR,
        "sensor_features.joblib"
    )
)

print("\nTraining Complete")
