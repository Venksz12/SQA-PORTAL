import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "supplier_manufacturing_data.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

df = pd.read_csv(CSV_PATH)

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "supplier_risk.joblib"
    )
)

features = joblib.load(
    os.path.join(
        MODEL_DIR,
        "supplier_features.joblib"
    )
)

X = df[features].fillna(0)

df["predicted_supplier_score"] = model.predict(X)

df["risk_rank"] = (
    df["predicted_supplier_score"]
    .rank(ascending=True)
)

output_file = os.path.join(
    BASE_DIR,
    "supplier_predictions.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nTop 20 Risky Suppliers\n")

cols = [
    "supplier_id",
    "predicted_supplier_score",
    "risk_rank",
    "ppm",
    "audit_score",
    "cpk"
]

print(
    df.sort_values(
        "predicted_supplier_score"
    )[cols].head(20)
)

print(
    f"\nSaved: {output_file}"
)
