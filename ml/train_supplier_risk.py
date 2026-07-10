import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Supplier Dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df)}")

TARGET = "supplier_score"

FEATURES = [
    "ppm",
    "audit_score",
    "cpk",
    "claim_count",
    "repeat_failure_rate",
    "corrective_actions",
    "supplier_response_days",
    "replacement_cost"
]

X = df[FEATURES].copy()
y = df[TARGET].copy()

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Supplier Risk Model...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

preds = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    preds
)

mse = mean_squared_error(
    y_test,
    preds
)

r2 = r2_score(
    y_test,
    preds
)

print("\nModel Performance")

print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R2  : {r2:.4f}")

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "supplier_risk.joblib"
    )
)

joblib.dump(
    FEATURES,
    os.path.join(
        MODEL_DIR,
        "supplier_features.joblib"
    )
)

print("\nSaved:")

print("supplier_risk.joblib")
print("supplier_features.joblib")