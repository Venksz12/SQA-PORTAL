import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "dealer_claims_10000.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

print("Loading Claims Dataset...")

df = pd.read_csv(CSV_PATH)

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "claim_fraud.joblib"
    )
)

feature_cols = joblib.load(
    os.path.join(
        MODEL_DIR,
        "claim_features.joblib"
    )
)

X = df.drop(
    columns=[
        "claim_id",
        "fraud_label"
    ],
    errors="ignore"
)

X = pd.get_dummies(X)

for col in feature_cols:
    if col not in X.columns:
        X[col] = 0

X = X[feature_cols]

print("Generating Fraud Predictions...")

df["fraud_prediction"] = model.predict(X)

df["fraud_probability"] = (
    model.predict_proba(X)[:, 1] * 100
)

output_file = os.path.join(
    BASE_DIR,
    "claim_fraud_predictions.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nTop Suspicious Claims\n")

top_claims = df.sort_values(
    "fraud_probability",
    ascending=False
)

print(
    top_claims[
        [
            "claim_id",
            "supplier_id",
            "claim_amount",
            "claim_frequency",
            "fraud_probability"
        ]
    ].head(20)
)

print(
    f"\nSaved: {output_file}"
)