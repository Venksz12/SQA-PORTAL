# train_sqa_models.py

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

DATA_PATH = "updates.csv"

# 1) Load dataset
df = pd.read_csv(DATA_PATH)

# 2) Create date features
if "test_date" in df.columns:
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df["test_month"] = df["test_date"].dt.month
    df["test_quarter"] = df["test_date"].dt.quarter
    df["test_dayofweek"] = df["test_date"].dt.dayofweek

# 3) Define targets
regression_target = "supplier_quality_score_recomputed_0_1"
classification_target = "risk_label_current_scenario"

# 4) Remove leakage columns
leakage_cols = [
    "record_id",
    "test_date",
    "part_quality_score_recomputed_0_1",
    "supplier_quality_score_recomputed_0_1",
    "sqm_gap_to_target_0_1",
    "part_gap_to_target_0_1",
    "weighted_loss_priority_0_1",
    "sqm_status",
    "action_priority",
    "risk_label_current_scenario",
    "dataset_note_current_scenario",
]

feature_cols = [c for c in df.columns if c not in leakage_cols]

X = df[feature_cols].copy()
y_reg = df[regression_target].copy()
y_clf = df[classification_target].copy()

# 5) Split numeric / categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

# 6) Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ---------------- REGRESSION MODEL ----------------
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ))
])

reg_model.fit(X_train_r, y_train_r)
reg_preds = reg_model.predict(X_test_r)

reg_metrics = {
    "MAE": float(mean_absolute_error(y_test_r, reg_preds)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test_r, reg_preds))),
    "R2": float(r2_score(y_test_r, reg_preds)),
}

# ---------------- CLASSIFICATION MODEL ----------------
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))
])

clf_model.fit(X_train_c, y_train_c)
clf_preds = clf_model.predict(X_test_c)

clf_metrics = {
    "Accuracy": float(accuracy_score(y_test_c, clf_preds)),
    "ConfusionMatrix": confusion_matrix(
        y_test_c, clf_preds, labels=sorted(y_clf.unique())
    ).tolist(),
    "LabelsOrder": sorted(y_clf.unique()),
    "ClassificationReport": classification_report(
        y_test_c, clf_preds, output_dict=True
    ),
}

# 7) Save models
joblib.dump(reg_model, "sqa_supplier_score_regressor.joblib")
joblib.dump(clf_model, "sqa_risk_classifier.joblib")

# 8) Save sample predictions
sample_output = X_test_r.head(10).copy()
sample_output["actual_supplier_quality_score"] = y_test_r.head(10).values
sample_output["predicted_supplier_quality_score"] = reg_model.predict(X_test_r.head(10))
sample_output.to_csv("sqa_sample_predictions.csv", index=False)

# 9) Save metrics
metrics = {
    "regression_metrics": reg_metrics,
    "classification_metrics": clf_metrics,
    "feature_count": len(feature_cols),
    "numeric_feature_count": len(numeric_features),
    "categorical_feature_count": len(categorical_features),
}

with open("sqa_training_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Training complete!")
print("Regression metrics:", reg_metrics)
print("Classification accuracy:", clf_metrics["Accuracy"])