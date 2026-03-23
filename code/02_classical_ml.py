"""
Phase 2 — Classical ML Baseline Pipeline
CS 464 Stroke Prediction Project

Preprocessing + model training + evaluation for all classical ML models.
Run:  python code/02_classical_ml.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
DATA_PATH = os.path.join("data", "healthcare-dataset-stroke-data.csv")
OUT_DIR = os.path.join("outputs", "results", "classical_ml_baseline_results")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 300
SEED = 42
N_FOLDS = 5

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# ────────────────────────────────────────────
# 1. Load & stateless cleaning
# ────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — CLASSICAL ML BASELINE PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Convert bmi to numeric (handles "N/A" strings)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# Create bmi_missing indicator BEFORE imputation
df["bmi_missing"] = df["bmi"].isna().astype(int)

# Keep id aside for traceability, then drop from features
patient_ids = df["id"].copy()
df.drop(columns=["id"], inplace=True)

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Stroke=1: {df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)")
print(f"Stroke=0: {(df['stroke']==0).sum()} ({(df['stroke']==0).mean()*100:.2f}%)")

# ────────────────────────────────────────────
# 2. Define feature groups
# ────────────────────────────────────────────
NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
PASSTHROUGH = ["hypertension", "heart_disease", "bmi_missing"]

ALL_FEATURES = NUMERIC + CATEGORICAL + PASSTHROUGH

X = df[ALL_FEATURES]
y = df["stroke"]

# ────────────────────────────────────────────
# 3. Stratified 80/20 train/test split
# ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"\nTrain: {X_train.shape[0]} rows (stroke={y_train.sum()})")
print(f"Test:  {X_test.shape[0]} rows (stroke={y_test.sum()})")

# ────────────────────────────────────────────
# 4. Preprocessing via ColumnTransformer
# ────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("pass", "passthrough", PASSTHROUGH),
    ]
)

# ────────────────────────────────────────────
# 5. Define models
# ────────────────────────────────────────────
# Compute scale_pos_weight for XGBoost
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw = n_neg / n_pos

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED
    ),
    "Gaussian NB": GaussianNB(),
    "KNN k=1": KNeighborsClassifier(n_neighbors=1),
    "KNN k=3": KNeighborsClassifier(n_neighbors=3),
    "KNN k=5": KNeighborsClassifier(n_neighbors=5),
    "KNN k=7": KNeighborsClassifier(n_neighbors=7),
    "KNN k=9": KNeighborsClassifier(n_neighbors=9),
    "SVM (RBF)": SVC(
        class_weight="balanced", probability=True, random_state=SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, scale_pos_weight=spw,
        random_state=SEED, eval_metric="logloss", verbosity=0,
    ),
}

# ────────────────────────────────────────────
# 6. Build pipelines (Preprocessor → SMOTE → Model)
# ────────────────────────────────────────────
pipelines = {}
for name, model in models.items():
    pipelines[name] = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])

# ────────────────────────────────────────────
# 7. 5-fold stratified CV on training set
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cv_records = []

for name, pipe in pipelines.items():
    print(f"\n  Training: {name} ...", end=" ", flush=True)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }

    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )

    record = {"Model": name}
    for metric_key, score_key in [
        ("Accuracy", "test_accuracy"),
        ("F1", "test_f1"),
        ("Recall", "test_recall"),
        ("Precision", "test_precision"),
        ("AUC-ROC", "test_roc_auc"),
        ("AUC-PR", "test_avg_precision"),
    ]:
        vals = scores[score_key]
        record[f"{metric_key} (mean)"] = f"{vals.mean():.4f}"
        record[f"{metric_key} (std)"] = f"{vals.std():.4f}"

    cv_records.append(record)
    print(f"AUC-PR={float(record['AUC-PR (mean)']):.4f} ± {float(record['AUC-PR (std)']):.4f}")

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(os.path.join(OUT_DIR, "cv_results.csv"), index=False)
print(f"\n  → Saved cv_results.csv")

# ────────────────────────────────────────────
# 8. Final evaluation on held-out test set
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 60)

test_records = []
roc_data = {}
pr_data = {}
cm_data = {}

for name, pipe in pipelines.items():
    print(f"\n  Evaluating: {name} ...", end=" ", flush=True)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    test_records.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "F1": f"{f1:.4f}",
        "Recall": f"{rec:.4f}",
        "Precision": f"{prec:.4f}",
        "AUC-ROC": f"{roc:.4f}",
        "AUC-PR": f"{ap:.4f}",
    })

    # Store for plotting
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, roc)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_data[name] = (rec_curve, prec_curve, ap)

    cm_data[name] = cm

    print(f"AUC-PR={ap:.4f}, F1={f1:.4f}, Recall={rec:.4f}")

test_df = pd.DataFrame(test_records)
test_df.to_csv(os.path.join(OUT_DIR, "test_results.csv"), index=False)
print(f"\n  → Saved test_results.csv")

# Print summary table
print("\n" + "-" * 80)
print(test_df.to_string(index=False))
print("-" * 80)

# ────────────────────────────────────────────
# Plot styling — distinct colors & line styles
# ────────────────────────────────────────────
MODEL_STYLES = {
    "Logistic Regression": {"color": "#2563eb", "ls": "-",  "lw": 2.5},
    "Gaussian NB":         {"color": "#dc2626", "ls": "-",  "lw": 2.0},
    "KNN k=1":             {"color": "#a3a3a3", "ls": ":",  "lw": 1.3},
    "KNN k=3":             {"color": "#a3a3a3", "ls": "-.", "lw": 1.3},
    "KNN k=5":             {"color": "#a3a3a3", "ls": "--", "lw": 1.3},
    "KNN k=7":             {"color": "#737373", "ls": "-.", "lw": 1.3},
    "KNN k=9":             {"color": "#737373", "ls": "--", "lw": 1.3},
    "SVM (RBF)":           {"color": "#7c3aed", "ls": "-",  "lw": 2.0},
    "Random Forest":       {"color": "#16a34a", "ls": "-",  "lw": 2.0},
    "XGBoost":             {"color": "#ea580c", "ls": "-",  "lw": 2.5},
}

# ────────────────────────────────────────────
# 9. ROC Curves (sorted by AUC descending)
# ────────────────────────────────────────────
print("\n[Plot] ROC Curves")

fig, ax = plt.subplots(figsize=(9, 7))

sorted_roc = sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True)
for name, (fpr, tpr, auc_val) in sorted_roc:
    s = MODEL_STYLES[name]
    ax.plot(fpr, tpr, color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
            label=f"{name} (AUC={auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Classical ML Baselines")
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_aspect("equal")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_curves.png"))
plt.close()
print("  → Saved roc_curves.png")

# ────────────────────────────────────────────
# 10. Precision-Recall Curves (sorted by AP descending)
# ────────────────────────────────────────────
print("[Plot] Precision-Recall Curves")

fig, ax = plt.subplots(figsize=(9, 7))

baseline_pr = y_test.mean()
sorted_pr = sorted(pr_data.items(), key=lambda x: x[1][2], reverse=True)
for name, (rec_c, prec_c, ap_val) in sorted_pr:
    s = MODEL_STYLES[name]
    ax.plot(rec_c, prec_c, color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
            label=f"{name} (AP={ap_val:.3f})")

ax.axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.3, linewidth=1,
           label=f"Baseline (prevalence={baseline_pr:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Classical ML Baselines")
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_aspect("equal")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pr_curves.png"))
plt.close()
print("  → Saved pr_curves.png")

# ────────────────────────────────────────────
# 11. Confusion Matrices (normalized + raw counts)
# ────────────────────────────────────────────
print("[Plot] Confusion Matrices")

n_models = len(cm_data)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
axes = axes.flatten()

for i, (name, cm) in enumerate(cm_data.items()):
    ax = axes[i]
    # Normalize each row to percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Annotation: percentage + raw count
    annot = np.array([
        [f"{cm_norm[r, c]:.1f}%\n(n={cm[r, c]})" for c in range(2)]
        for r in range(2)
    ])

    # Use imshow for equal cell sizes (no aspect distortion)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")

    # Add annotations
    for r in range(2):
        for c in range(2):
            text_color = "white" if cm_norm[r, c] > 60 else "black"
            ax.text(c, r, annot[r, c], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Stroke", "Stroke"], fontsize=10)
    ax.set_yticklabels(["No Stroke", "Stroke"], fontsize=10)
    ax.set_title(name, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)

# hide unused subplots
for j in range(n_models, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Confusion Matrices (Row-Normalized %) — Classical ML Baselines",
             fontsize=16, fontweight="bold", y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"))
plt.close()
print("  → Saved confusion_matrices.png")

# ════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — Classical ML Baselines")
print(f"  All outputs saved to: {OUT_DIR}/")
print("=" * 60)
