"""
Phase 3 — Imbalance Strategy Experiments
CS 464 Stroke Prediction Project

Compares 7 imbalance-handling strategies across 3 model families.
Isolates resampling from class-weighting to avoid double-correction.

Run:  python code/03_imbalance_experiments.py
"""

import os
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
DATA_PATH = os.path.join("data", "healthcare-dataset-stroke-data.csv")
OUT_DIR = os.path.join("outputs", "results", "imbalance_experiments")
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
# 1. Load & stateless cleaning (identical to 02)
# ────────────────────────────────────────────
print("=" * 60)
print("PHASE 3 — IMBALANCE STRATEGY EXPERIMENTS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi_missing"] = df["bmi"].isna().astype(int)
df.drop(columns=["id"], inplace=True)

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Stroke=1: {df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)")

# ────────────────────────────────────────────
# 2. Feature groups & split (identical to 02)
# ────────────────────────────────────────────
NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
PASSTHROUGH = ["hypertension", "heart_disease", "bmi_missing"]

X = df[NUMERIC + CATEGORICAL + PASSTHROUGH]
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"Train: {X_train.shape[0]} rows (stroke={y_train.sum()})")
print(f"Test:  {X_test.shape[0]} rows (stroke={y_test.sum()})")

# ────────────────────────────────────────────
# 3. Preprocessor (identical to 02)
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
# 4. Experiment grid: models × strategies
# ────────────────────────────────────────────
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw = n_neg / n_pos  # ~19.5

# Model factories — class_weight controlled per strategy
def make_models(use_class_weight: bool):
    """Return dict of models. If use_class_weight=True, enable built-in weighting."""
    cw = "balanced" if use_class_weight else None
    sw = spw if use_class_weight else 1
    return {
        "Logistic Regression": LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=sw,
            random_state=SEED, eval_metric="logloss", verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=cw, random_state=SEED
        ),
    }

# Strategies: (name, resampler_or_None, use_class_weight)
STRATEGIES = [
    ("No Resampling",       None,                                              False),
    ("Class Weight Only",   None,                                              True),
    ("SMOTE",               SMOTE(random_state=SEED),                          False),
    ("ADASYN",              ADASYN(random_state=SEED),                         False),
    ("BorderlineSMOTE",     BorderlineSMOTE(random_state=SEED),                False),
    ("SMOTE + Tomek",       SMOTETomek(random_state=SEED),                     False),
    ("SMOTE + ENN",         SMOTEENN(random_state=SEED),                       False),
]

MODEL_NAMES = ["Logistic Regression", "XGBoost", "Random Forest"]

print(f"\nExperiment grid: {len(MODEL_NAMES)} models × {len(STRATEGIES)} strategies = {len(MODEL_NAMES) * len(STRATEGIES)} experiments")

# ────────────────────────────────────────────
# 5. Run 5-fold stratified CV for all combos
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

cv_records = []
total_experiments = len(MODEL_NAMES) * len(STRATEGIES)
exp_count = 0

for strat_name, resampler, use_cw in STRATEGIES:
    models = make_models(use_cw)
    for model_name in MODEL_NAMES:
        exp_count += 1
        model = models[model_name]
        label = f"[{exp_count}/{total_experiments}] {model_name} + {strat_name}"
        print(f"\n  {label} ...", end=" ", flush=True)

        # Build pipeline
        steps = [("preprocessor", preprocessor)]
        if resampler is not None:
            # Clone resampler to avoid state leakage between models
            from sklearn.base import clone as sk_clone
            steps.append(("resampler", sk_clone(resampler)))
        steps.append(("model", model))
        pipe = ImbPipeline(steps)

        # Manual CV to collect probabilities for AUC metrics
        fold_metrics = {k: [] for k in ["accuracy", "f1", "recall", "precision", "roc_auc", "auc_pr"]}

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            try:
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_val)
                y_prob = pipe.predict_proba(X_val)[:, 1]

                fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
                fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
                fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
                fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
                fold_metrics["auc_pr"].append(average_precision_score(y_val, y_prob))
            except Exception as e:
                print(f"\n    ⚠ Fold {fold_idx} failed: {e}")
                # Append NaN so we can still compute stats
                for k in fold_metrics:
                    fold_metrics[k].append(np.nan)

        record = {"Model": model_name, "Strategy": strat_name}
        for metric_key, display_key in [
            ("accuracy", "Accuracy"), ("f1", "F1"), ("recall", "Recall"),
            ("precision", "Precision"), ("roc_auc", "AUC-ROC"), ("auc_pr", "AUC-PR"),
        ]:
            vals = np.array(fold_metrics[metric_key])
            record[f"{display_key} (mean)"] = np.nanmean(vals)
            record[f"{display_key} (std)"] = np.nanstd(vals)

        cv_records.append(record)
        print(f"AUC-PR={record['AUC-PR (mean)']:.4f} ± {record['AUC-PR (std)']:.4f}")

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(os.path.join(OUT_DIR, "imbalance_cv_results.csv"), index=False, float_format="%.4f")
print(f"\n  → Saved imbalance_cv_results.csv")

# ────────────────────────────────────────────
# 6. Test set evaluation for all combos
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 60)

test_records = []
# Store curves for best strategy per model
best_per_model = {}  # model_name -> (strat_name, auc_pr)

roc_curves_data = {}
pr_curves_data = {}

exp_count = 0
for strat_name, resampler, use_cw in STRATEGIES:
    models = make_models(use_cw)
    for model_name in MODEL_NAMES:
        exp_count += 1
        model = models[model_name]
        label = f"[{exp_count}/{total_experiments}] {model_name} + {strat_name}"
        print(f"\n  {label} ...", end=" ", flush=True)

        steps = [("preprocessor", preprocessor)]
        if resampler is not None:
            from sklearn.base import clone as sk_clone
            steps.append(("resampler", sk_clone(resampler)))
        steps.append(("model", model))
        pipe = ImbPipeline(steps)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        test_records.append({
            "Model": model_name, "Strategy": strat_name,
            "Accuracy": acc, "F1": f1, "Recall": rec,
            "Precision": prec, "AUC-ROC": roc, "AUC-PR": ap,
        })

        # Store curve data
        combo_key = f"{model_name} + {strat_name}"
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[combo_key] = (fpr, tpr, roc)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_curves_data[combo_key] = (rec_c, prec_c, ap)

        # Track best strategy per model
        if model_name not in best_per_model or ap > best_per_model[model_name][1]:
            best_per_model[model_name] = (strat_name, ap)

        print(f"AUC-PR={ap:.4f}, F1={f1:.4f}, Recall={rec:.4f}")

test_df = pd.DataFrame(test_records)
test_df.to_csv(os.path.join(OUT_DIR, "imbalance_test_results.csv"), index=False, float_format="%.4f")
print(f"\n  → Saved imbalance_test_results.csv")

# Print summary table
print("\n" + "-" * 90)
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("-" * 90)

# ────────────────────────────────────────────
# 7. Heatmap: AUC-PR (models × strategies)
# ────────────────────────────────────────────
print("\n[Plot] AUC-PR Heatmap")

# Create pivot tables for both CV and test
heatmap_cv = cv_df.pivot(index="Model", columns="Strategy", values="AUC-PR (mean)")
heatmap_test = test_df.pivot(index="Model", columns="Strategy", values="AUC-PR")

# Reorder rows and columns
strat_order = [s[0] for s in STRATEGIES]
model_order = MODEL_NAMES
heatmap_cv = heatmap_cv.loc[model_order, strat_order]
heatmap_test = heatmap_test.loc[model_order, strat_order]

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# CV heatmap
sns.heatmap(
    heatmap_cv, annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.8, linecolor="white", ax=axes[0],
    cbar_kws={"label": "AUC-PR (mean CV)"},
    vmin=heatmap_cv.min().min() * 0.9,
    vmax=heatmap_cv.max().max() * 1.02,
)
axes[0].set_title("AUC-PR — 5-Fold CV (mean)", fontsize=14, fontweight="bold", pad=12)
axes[0].set_ylabel("")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=30)

# Test heatmap
sns.heatmap(
    heatmap_test, annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.8, linecolor="white", ax=axes[1],
    cbar_kws={"label": "AUC-PR (test)"},
    vmin=heatmap_test.min().min() * 0.9,
    vmax=heatmap_test.max().max() * 1.02,
)
axes[1].set_title("AUC-PR — Held-Out Test Set", fontsize=14, fontweight="bold", pad=12)
axes[1].set_ylabel("")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=30)

# Mark best per row (model)
for ax_idx, hmap in enumerate([heatmap_cv, heatmap_test]):
    ax = axes[ax_idx]
    for row_idx, model in enumerate(model_order):
        best_col_idx = hmap.loc[model].values.argmax()
        ax.add_patch(plt.Rectangle(
            (best_col_idx, row_idx), 1, 1,
            fill=False, edgecolor="black", linewidth=3
        ))

fig.suptitle("Imbalance Strategy Comparison — AUC-PR", fontsize=16, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "imbalance_heatmap.png"))
plt.close()
print("  → Saved imbalance_heatmap.png")

# ────────────────────────────────────────────
# 8. PR Curves — best strategy per model
# ────────────────────────────────────────────
print("[Plot] PR Curves (best strategy per model)")

COLORS = {
    "Logistic Regression": "#2563eb",
    "XGBoost": "#ea580c",
    "Random Forest": "#16a34a",
}

fig, ax = plt.subplots(figsize=(9, 7))
baseline_pr = y_test.mean()

# Sort by AUC-PR descending
sorted_best = sorted(best_per_model.items(), key=lambda x: x[1][1], reverse=True)

for model_name, (strat_name, _) in sorted_best:
    combo_key = f"{model_name} + {strat_name}"
    rec_c, prec_c, ap_val = pr_curves_data[combo_key]
    ax.plot(rec_c, prec_c, color=COLORS[model_name], linewidth=2.5,
            label=f"{model_name} + {strat_name} (AP={ap_val:.3f})")

ax.axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.3, linewidth=1,
           label=f"Baseline (prevalence={baseline_pr:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Best Strategy per Model", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_aspect("equal")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "imbalance_pr_curves.png"))
plt.close()
print("  → Saved imbalance_pr_curves.png")

# ────────────────────────────────────────────
# 9. ROC Curves — best strategy per model
# ────────────────────────────────────────────
print("[Plot] ROC Curves (best strategy per model)")

fig, ax = plt.subplots(figsize=(9, 7))

for model_name, (strat_name, _) in sorted_best:
    combo_key = f"{model_name} + {strat_name}"
    fpr, tpr, auc_val = roc_curves_data[combo_key]
    ax.plot(fpr, tpr, color=COLORS[model_name], linewidth=2.5,
            label=f"{model_name} + {strat_name} (AUC={auc_val:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Best Strategy per Model", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_aspect("equal")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "imbalance_roc_curves.png"))
plt.close()
print("  → Saved imbalance_roc_curves.png")

# ────────────────────────────────────────────
# 10. Summary: best strategy per model
# ────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEST STRATEGY PER MODEL (by test AUC-PR)")
print("=" * 60)

for model_name, (strat_name, ap) in sorted_best:
    row = test_df[(test_df["Model"] == model_name) & (test_df["Strategy"] == strat_name)].iloc[0]
    print(f"\n  {model_name}:")
    print(f"    Strategy  = {strat_name}")
    print(f"    AUC-PR    = {row['AUC-PR']:.4f}")
    print(f"    AUC-ROC   = {row['AUC-ROC']:.4f}")
    print(f"    Recall    = {row['Recall']:.4f}")
    print(f"    F1        = {row['F1']:.4f}")
    print(f"    Precision = {row['Precision']:.4f}")

# Compare with baseline
print("\n" + "-" * 60)
print("COMPARISON WITH PHASE 2 BASELINES")
print("-" * 60)
print(f"  Phase 2 best: Logistic Regression + SMOTE + class_weight")
print(f"    AUC-PR = 0.2665, AUC-ROC = 0.8445")
print(f"\n  Phase 3 best: {sorted_best[0][0]} + {sorted_best[0][1][0]}")
print(f"    AUC-PR = {sorted_best[0][1][1]:.4f}")

delta = sorted_best[0][1][1] - 0.2665
if delta > 0:
    print(f"    Δ AUC-PR = +{delta:.4f} ✅ improvement")
elif delta < 0:
    print(f"    Δ AUC-PR = {delta:.4f} ⚠ regression")
else:
    print(f"    Δ AUC-PR = 0 (no change)")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE — Imbalance Strategy Experiments")
print(f"  All outputs saved to: {OUT_DIR}/")
print("=" * 60)
