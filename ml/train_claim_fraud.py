import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

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

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Claims Dataset...")

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))

TARGET = "fraud_label"

DROP_COLS = [
    "claim_id"
]

X = df.drop(
    columns=DROP_COLS + [TARGET]
)

X = pd.get_dummies(X)

y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Fraud Model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print("\nAccuracy:")
print(acc)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        preds
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        preds
    )
)

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "claim_fraud.joblib"
    )
)

joblib.dump(
    X.columns.tolist(),
    os.path.join(
        MODEL_DIR,
        "claim_features.joblib"
    )
)

print("\nSaved:")

print("claim_fraud.joblib")
print("claim_features.joblib")