import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Get directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
np.random.seed(42)

print("All libraries loaded successfully")
print(f"Numpy: {np.__version__} | Pandas: {pd.__version__}")

# ===============================
# Classification Dataset
# ===============================
cancer = load_breast_cancer()
X_clf, y_clf = cancer.data, cancer.target  # 0 = malignant, 1 = benign

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)

scaler_c = StandardScaler()
X_train_c = scaler_c.fit_transform(X_train_c)
X_test_c = scaler_c.transform(X_test_c)

# ===============================
# Regression Dataset
# ===============================
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

# ===============================
# Quick Data Summary
# ===============================
print("\nCLASSIFICATION DATASET: Breast Cancer")
print(f"Train: {X_train_c.shape} | Test: {X_test_c.shape}")

malignant = (y_clf == 0).sum()
benign = (y_clf == 1).sum()

print(f"Class balance - Malignant: {malignant} | Benign: {benign}")

print("\nREGRESSION DATASET: Diabetes")
print(f"Train: {X_train_r.shape} | Test: {X_test_r.shape}")
print(f"Target range: [{y_reg.min():.0f}, {y_reg.max():.0f}]")

# ===============================
# Feature Names
# ===============================
print("\nFirst 5 feature names:")
print(cancer.feature_names[:5])

# ===============================
# Imbalance Ratio
# ===============================
imbalance_ratio = malignant / benign
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")

# ===============================
# Class Distribution Bar Chart
# ===============================
plt.figure(figsize=(6, 4))

classes = ["Malignant", "Benign"]
counts = [malignant, benign]

bars = plt.bar(classes, counts, color=["tab:blue", "tab:orange"])

for bar, val in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(val),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.title("Class Distribution - Breast Cancer Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig(os.path.join(script_dir, "class_distribution.png"), bbox_inches="tight")
plt.show()

# ===============================
# Train Classification Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_c, y_train_c)

    y_pred = model.predict(X_test_c)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_c)[:, 1]
    else:
        y_proba = None

    cm = confusion_matrix(y_test_c, y_pred)
    acc = accuracy_score(y_test_c, y_pred)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "cm": cm
    }

    print(f"{name}: Accuracy = {acc:.4f}")

# ===============================
# Confusion Matrices Side by Side
# ===============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res["cm"],
        display_labels=cancer.target_names
    )

    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc: {res['accuracy']:.3f}", fontsize=11)

plt.suptitle(
    "Confusion Matrices - All Models",
    y=1.05,
    fontsize=13,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "confusion_matrices.png"), bbox_inches="tight")
plt.show()

# ===============================
# Manual Accuracy Calculation
# Logistic Regression
# ===============================
log_reg_cm = results["Logistic Regression"]["cm"]

print("\nLogistic Regression Confusion Matrix:")
print(log_reg_cm)

TN = log_reg_cm[0][0]
FP = log_reg_cm[0][1]
FN = log_reg_cm[1][0]
TP = log_reg_cm[1][1]

print(f"\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

manual_acc = (TP + TN) / (TP + TN + FP + FN)
sklearn_acc = results["Logistic Regression"]["accuracy"]

print(f"\nManual Accuracy: {manual_acc:.4f}")
print(f"Sklearn Accuracy: {sklearn_acc:.4f}")