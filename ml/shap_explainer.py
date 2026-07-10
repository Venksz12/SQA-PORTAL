import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.savefig(
    os.path.join(
        BASE_DIR,
        "supplier_shap_summary.png"
    ),
    bbox_inches="tight"
)

print("Saved supplier_shap_summary.png")