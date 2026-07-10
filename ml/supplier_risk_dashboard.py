import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "supplier_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dashboard_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))

# -----------------------------
# Top 20 Risky Suppliers
# -----------------------------

top20 = df.sort_values(
    "predicted_supplier_score"
).head(20)

top20.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top20_risky_suppliers.csv"
    ),
    index=False
)

# -----------------------------
# Risk Distribution
# -----------------------------

plt.figure(figsize=(10,6))

plt.hist(
    df["predicted_supplier_score"],
    bins=30
)

plt.title(
    "Supplier Risk Distribution"
)

plt.xlabel(
    "Predicted Supplier Score"
)

plt.ylabel(
    "Count"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "risk_distribution.png"
    )
)

plt.close()

# -----------------------------
# PPM vs Supplier Score
# -----------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    df["ppm"],
    df["predicted_supplier_score"],
    alpha=0.5
)

plt.title(
    "PPM vs Predicted Supplier Score"
)

plt.xlabel("PPM")

plt.ylabel(
    "Predicted Supplier Score"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ppm_vs_score.png"
    )
)

plt.close()

# -----------------------------
# Audit Score vs Supplier Score
# -----------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    df["audit_score"],
    df["predicted_supplier_score"],
    alpha=0.5
)

plt.title(
    "Audit Score vs Predicted Supplier Score"
)

plt.xlabel(
    "Audit Score"
)

plt.ylabel(
    "Predicted Supplier Score"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "audit_vs_score.png"
    )
)

plt.close()

print("\nDashboard Files Generated:\n")

print(
    "top20_risky_suppliers.csv"
)

print(
    "risk_distribution.png"
)

print(
    "ppm_vs_score.png"
)

print(
    "audit_vs_score.png"
)