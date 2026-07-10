import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.model_selection import (
    learning_curve,
    cross_val_score
)

from sklearn.calibration import calibration_curve

import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# GENERIC EVALUATION FUNCTION
# =========================================================

def save_roc_curve(
        y_true,
        y_prob,
        filename
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        color="blue",
        linewidth=2,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="red"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()

def save_pr_curve(
        y_true,
        y_prob,
        filename
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    plt.figure(figsize=(8,6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        color="green"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title(
        "Precision Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_learning_curve(
        model,
        X,
        y,
        filename
):

    train_sizes, train_scores, test_scores = (
        learning_curve(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
    )

    train_mean = np.mean(
        train_scores,
        axis=1
    )

    test_mean = np.mean(
        test_scores,
        axis=1
    )

    plt.figure(figsize=(8,6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training"
    )

    plt.plot(
        train_sizes,
        test_mean,
        marker="o",
        label="Validation"
    )

    plt.xlabel(
        "Training Samples"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Learning Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_cv_plot(
        model,
        X,
        y,
        filename
):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    plt.figure(figsize=(8,6))

    sns.barplot(
        x=list(range(
            1,
            len(scores)+1
        )),
        y=scores
    )

    plt.title(
        f"CV Mean={scores.mean():.4f}"
    )

    plt.xlabel(
        "Fold"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_feature_importance(
        model,
        feature_names,
        filename
):

    importance = (
        model.feature_importances_
    )

    imp_df = pd.DataFrame({

        "Feature":
        feature_names,

        "Importance":
        importance

    })

    imp_df = imp_df.sort_values(
        "Importance",
        ascending=False
    )

    plt.figure(
        figsize=(10,8)
    )

    sns.barplot(

        data=imp_df.head(20),

        x="Importance",

        y="Feature"

    )

    plt.title(
        "Top 20 Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_feature_importance(
        model,
        feature_names,
        filename
):

    importance = (
        model.feature_importances_
    )

    imp_df = pd.DataFrame({

        "Feature":
        feature_names,

        "Importance":
        importance

    })

    imp_df = imp_df.sort_values(
        "Importance",
        ascending=False
    )

    plt.figure(
        figsize=(10,8)
    )

    sns.barplot(

        data=imp_df.head(20),

        x="Importance",

        y="Feature"

    )

    plt.title(
        "Top 20 Important Features"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_error_distribution(
        y_true,
        y_pred,
        filename
):

    errors = (
        y_true -
        y_pred
    )

    plt.figure(
        figsize=(8,6)
    )

    sns.histplot(
        errors,
        bins=25,
        kde=True
    )

    plt.title(
        "Error Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_residual_plot(
        y_true,
        y_pred,
        filename
):

    residuals = (
        y_true -
        y_pred
    )

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color="red"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()
def save_residual_plot(
        y_true,
        y_pred,
        filename
):

    residuals = (
        y_true -
        y_pred
    )

    plt.figure(
        figsize=(8,6)
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color="red"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        filename
    )

    plt.close()

def evaluate_model(
        df,
        actual_col,
        pred_col,
        model_name,
        image_file,
        report_file,
        cmap="Blues"
):

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    y_true = df[actual_col]
    y_pred = df[pred_col]

    # -----------------------------------
    # Metrics
    # -----------------------------------

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -----------------------------------
    # Classification Report
    # -----------------------------------

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report\n")
    print(report)

    # -----------------------------------
    # Confusion Matrix
    # -----------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix\n")
    print(cm)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        image_file
    )

    plt.close()

    print(
        f"\nSaved Matrix -> {image_file}"
    )

    # -----------------------------------
    # Save Report
    # -----------------------------------

    with open(
            report_file,
            "w",
            encoding="utf-8"
    ) as f:

        f.write(
            f"MODEL : {model_name}\n\n"
        )

        f.write(
            f"Accuracy  : {acc:.4f}\n"
        )

        f.write(
            f"Precision : {prec:.4f}\n"
        )

        f.write(
            f"Recall    : {rec:.4f}\n"
        )

        f.write(
            f"F1 Score  : {f1:.4f}\n\n"
        )

        f.write("Confusion Matrix\n\n")
        f.write(str(cm))

        f.write("\n\n")

        f.write(report)

    print(
        f"Saved Report -> {report_file}"
    )


# =========================================================
# SENSOR MODEL
# =========================================================

try:

    sensor_df = pd.read_csv(
        "sensor_anomaly_results.csv"
    )

    print(
        "\nSensor Columns:"
    )

    print(
        sensor_df.columns.tolist()
    )

    if (
        "actual_label" in sensor_df.columns
        and
        "anomaly_flag" in sensor_df.columns
    ):

        evaluate_model(
            sensor_df,
            "actual_label",
            "anomaly_flag",
            "Sensor Anomaly Model",
            "sensor_confusion_matrix.png",
            "sensor_report.txt",
            "Blues"
        )

    else:

        print(
            "\nSensor CSV missing:"
        )

        print(
            "actual_label or anomaly_flag"
        )

except Exception as e:

    print(
        "\nSensor Evaluation Failed:"
    )

    print(e)

# =========================================================
# SUPPLIER MODEL
# =========================================================

try:

    supplier_df = pd.read_csv(
        "supplier_predictions.csv"
    )

    print(
        "\nSupplier Columns:"
    )

    print(
        supplier_df.columns.tolist()
    )

    if (
        "actual_risk" in supplier_df.columns
        and
        "predicted_risk" in supplier_df.columns
    ):

        evaluate_model(
            supplier_df,
            "actual_risk",
            "predicted_risk",
            "Supplier Risk Model",
            "supplier_confusion_matrix.png",
            "supplier_report.txt",
            "Greens"
        )

    else:

        print(
            "\nSupplier CSV missing:"
        )

        print(
            "actual_risk or predicted_risk"
        )

except Exception as e:

    print(
        "\nSupplier Evaluation Failed:"
    )

    print(e)

# =========================================================
# FRAUD MODEL
# =========================================================

try:

    fraud_df = pd.read_csv(
        "claim_fraud_predictions.csv"
    )

    print(
        "\nFraud Columns:"
    )

    print(
        fraud_df.columns.tolist()
    )

    if (
        "actual_fraud" in fraud_df.columns
        and
        "predicted_fraud" in fraud_df.columns
    ):

        evaluate_model(
            fraud_df,
            "actual_fraud",
            "predicted_fraud",
            "Claim Fraud Model",
            "fraud_confusion_matrix.png",
            "fraud_report.txt",
            "Reds"
        )

    else:

        print(
            "\nFraud CSV missing:"
        )

        print(
            "actual_fraud or predicted_fraud"
        )

except Exception as e:

    print(
        "\nFraud Evaluation Failed:"
    )

    print(e)

print("\n")
print("=" * 70)
print("ALL MODEL EVALUATIONS COMPLETED")
print("=" * 70)