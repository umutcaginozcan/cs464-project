"""
Winner Model Analysis — Logistic Regression (median_noflag + undersample)
CS 464 Stroke Prediction Project

Based on hyperparameter search results, the global best configuration is:
  Model       : Logistic Regression
  Imputation  : median_noflag (SimpleImputer median, NO bmi_missing flag)
  Resampling  : undersample  (RandomUnderSampler)
  Params      : penalty=l2, C=1, class_weight=None

This script:
  1. Retrains the winner pipeline from scratch
  2. Calibration curve (reliability diagram)
  3. Metrics on: Train, 5-fold stratified CV, Test
  4. SHAP global feature importance
  5. LIME local explanation for individual predictions

Outputs:
  outputs/figures/winner_analysis/  →  all plots
  outputs/results/winner_analysis/  →  all CSV tables
"""

import sys, os

# ── Path Setup ──────────────────────────────────────────────────────────────
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
os.chdir(_project_dir)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

import shap
import lime
import lime.lime_tabular

from shared import load_data, get_split, SEED, N_FOLDS, setup_plot_style

warnings.filterwarnings("ignore")
setup_plot_style()

# ── Output Directories ─────────────────────────────────────────────────────
FIG_DIR = os.path.join("outputs", "figures", "winner_analysis")
RES_DIR = os.path.join("outputs", "results", "winner_analysis")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ── Column groups (median_noflag → no bmi_missing flag) ─────────────────────
NUMERIC     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
PASSTHROUGH = ["hypertension", "heart_disease"]   # NO bmi_missing for this config

print("=" * 70)
print("WINNER MODEL ANALYSIS — Logistic Regression")
print("  Imputation : median_noflag")
print("  Resampling : undersample")
print("  Params     : penalty=l2, C=1")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA LOADING & SPLITTING (identical to hyper search)
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
X_train, X_test, y_train, y_test = get_split(df)

print(f"\nTrain: {len(X_train)} (stroke={y_train.sum()})")
print(f"Test:  {len(X_test)} (stroke={y_test.sum()})")

# Select only the columns this config uses (no bmi_missing)
FEATURES = NUMERIC + CATEGORICAL + PASSTHROUGH
X_train_w = X_train[FEATURES].copy()
X_test_w  = X_test[FEATURES].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD & TRAIN WINNER PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_winner_pipeline():
    """Build the exact winner pipeline from hyper search."""
    preprocessor = ColumnTransformer([
        ("num", ImbPipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("pass", "passthrough", PASSTHROUGH),
    ])
    
    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("resampler", RandomUnderSampler(random_state=SEED)),
        ("model", LogisticRegression(
            penalty="l2", C=1, max_iter=2000, solver="saga", random_state=SEED
        )),
    ])
    return pipe


pipe = build_winner_pipeline()
pipe.fit(X_train_w, y_train)

y_prob_test  = pipe.predict_proba(X_test_w)[:, 1]
y_pred_test  = pipe.predict(X_test_w)
y_prob_train = pipe.predict_proba(X_train_w)[:, 1]
y_pred_train = pipe.predict(X_train_w)

print("\n✅ Winner pipeline trained successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. METRICS: Train / 5-Fold CV / Test
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_prob, label=""):
    """Compute all relevant metrics and return as dict."""
    return {
        "Set":         label,
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, zero_division=0),
        "Recall":      recall_score(y_true, y_pred, zero_division=0),
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "F2":          fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "AUC-ROC":     roc_auc_score(y_true, y_prob),
        "AUC-PR":      average_precision_score(y_true, y_prob),
        "Brier Score":  brier_score_loss(y_true, y_prob),
        "Log Loss":    log_loss(y_true, y_prob),
    }


# ── Train Metrics ──
train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train, "Train")

# ── CV Metrics (5-fold, using cross_val_predict for unbiased estimates) ──
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# We need per-fold metrics, not just aggregated
cv_fold_records = []
for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train_w, y_train)):
    X_tr, X_val = X_train_w.iloc[tr_idx], X_train_w.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    fold_pipe = build_winner_pipeline()
    fold_pipe.fit(X_tr, y_tr)
    
    y_val_pred = fold_pipe.predict(X_val)
    y_val_prob = fold_pipe.predict_proba(X_val)[:, 1]
    
    fold_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, f"CV Fold {fold_idx+1}")
    cv_fold_records.append(fold_metrics)

cv_fold_df = pd.DataFrame(cv_fold_records)

# Aggregate CV
cv_agg = {"Set": "CV (mean ± std)"}
for col in ["Accuracy", "Precision", "Recall", "F1", "F2", "AUC-ROC", "AUC-PR", "Brier Score", "Log Loss"]:
    vals = cv_fold_df[col].values
    cv_agg[col] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

# ── Test Metrics ──
test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test, "Test")

# ── Combine & Save ──
summary_rows = [train_metrics, test_metrics]
summary_df = pd.DataFrame(summary_rows)

# Add CV aggregate row (as strings for ± formatting)
cv_agg_row = pd.DataFrame([cv_agg])

print("\n" + "=" * 70)
print("METRICS SUMMARY")
print("=" * 70)
print("\n── Train ──")
for k, v in train_metrics.items():
    if k != "Set":
        print(f"  {k:>12}: {v:.4f}")

print("\n── 5-Fold CV ──")
for k, v in cv_agg.items():
    if k != "Set":
        print(f"  {k:>12}: {v}")

print("\n── Test ──")
for k, v in test_metrics.items():
    if k != "Set":
        print(f"  {k:>12}: {v:.4f}")

# Save
summary_df.to_csv(os.path.join(RES_DIR, "metrics_train_test.csv"), index=False, float_format="%.4f")
cv_fold_df.to_csv(os.path.join(RES_DIR, "metrics_cv_folds.csv"), index=False, float_format="%.4f")
cv_agg_row.to_csv(os.path.join(RES_DIR, "metrics_cv_aggregate.csv"), index=False)
print(f"\n✅ Metrics saved to {RES_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION CURVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Calibration Curve ──")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 3a. Reliability diagram ---
ax = axes[0]
fraction_pos, mean_pred = calibration_curve(y_test, y_prob_test, n_bins=10, strategy="uniform")

ax.plot(mean_pred, fraction_pos, "s-", color="#2563eb", linewidth=2.5,
        markersize=8, label="Winner (LR)", zorder=3)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Perfectly Calibrated")

ax.fill_between(mean_pred, fraction_pos, mean_pred,
                alpha=0.15, color="#2563eb")

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect("equal")

brier = brier_score_loss(y_test, y_prob_test)
ax.text(0.05, 0.92, f"Brier Score = {brier:.4f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# --- 3b. Probability distribution by class ---
ax2 = axes[1]
mask_0 = y_test == 0
mask_1 = y_test == 1

ax2.hist(y_prob_test[mask_0], bins=50, alpha=0.6, color="#3b82f6",
         label=f"No Stroke (n={mask_0.sum()})", density=True)
ax2.hist(y_prob_test[mask_1], bins=50, alpha=0.7, color="#ef4444",
         label=f"Stroke (n={mask_1.sum()})", density=True)

ax2.axvline(x=0.5, color="gray", ls=":", lw=1.5, alpha=0.6, label="Default t=0.5")
ax2.set_xlabel("Predicted Probability", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Predicted Probability Distribution by Class", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)

sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "calibration_curve.png"))
plt.close()
print(f"  Brier Score = {brier:.4f}")
print(f"  ✅ Saved: {FIG_DIR}/calibration_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFUSION MATRIX (Test Set)
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred_test)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(7, 6))
annot = np.array([[f"{cm_norm[r,c]:.1f}%\n(n={cm[r,c]})" for c in range(2)] for r in range(2)])
ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")
for r in range(2):
    for c in range(2):
        color = "white" if cm_norm[r, c] > 60 else "black"
        ax.text(c, r, annot[r, c], ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Stroke", "Stroke"], fontsize=12)
ax.set_yticklabels(["No Stroke", "Stroke"], fontsize=12)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Winner Model — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"))
plt.close()
print(f"  ✅ Saved: {FIG_DIR}/confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROC & PR CURVES (Test Set)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
auc_roc = roc_auc_score(y_test, y_prob_test)
axes[0].plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"AUC-ROC = {auc_roc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve", xlim=[0, 1], ylim=[0, 1.02])
axes[0].legend(fontsize=11)
axes[0].set_aspect("equal")

# PR
prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob_test)
auc_pr = average_precision_score(y_test, y_prob_test)
baseline = y_test.mean()
axes[1].plot(rec_c, prec_c, color="#2563eb", lw=2.5, label=f"AUC-PR = {auc_pr:.4f}")
axes[1].axhline(y=baseline, color="k", ls="--", alpha=0.3, lw=1,
                label=f"Prevalence ({baseline:.3f})")
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR Curve", xlim=[0, 1], ylim=[0, 1.02])
axes[1].legend(fontsize=11)
axes[1].set_aspect("equal")

sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"))
plt.close()
print(f"  ✅ Saved: {FIG_DIR}/roc_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP GLOBAL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SHAP Analysis ──")

# Get the fitted preprocessor and model from the pipeline
# Note: imblearn pipeline stores steps as list of (name, estimator)
fitted_preprocessor = pipe.named_steps["preprocessor"]
fitted_model = pipe.named_steps["model"]

# Transform the data (preprocessor only, no resampling for explanation)
X_train_transformed = fitted_preprocessor.transform(X_train_w)
X_test_transformed  = fitted_preprocessor.transform(X_test_w)

# Get feature names after preprocessing
num_features = NUMERIC
cat_features = list(fitted_preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
pass_features = PASSTHROUGH
all_feature_names = num_features + cat_features + pass_features

X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)
X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names)

# SHAP — LinearExplainer is exact for logistic regression
explainer = shap.LinearExplainer(fitted_model, X_train_df)
shap_values = explainer.shap_values(X_test_df)

# --- 6a. Summary plot (bar — global importance) ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, plot_type="bar",
                  show=False, max_display=20)
plt.title("SHAP Feature Importance (Global)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_global_bar.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {FIG_DIR}/shap_global_bar.png")

# --- 6b. Summary plot (dot — direction + magnitude) ---
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, plot_type="dot",
                  show=False, max_display=20)
plt.title("SHAP Summary (Direction + Magnitude)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_global_dot.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✅ Saved: {FIG_DIR}/shap_global_dot.png")

# --- Save SHAP importance values ---
shap_importance = pd.DataFrame({
    "Feature": all_feature_names,
    "Mean |SHAP|": np.abs(shap_values).mean(axis=0),
}).sort_values("Mean |SHAP|", ascending=False)
shap_importance.to_csv(os.path.join(RES_DIR, "shap_importance.csv"), index=False, float_format="%.4f")
print(f"  ✅ Saved: {RES_DIR}/shap_importance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. LIME LOCAL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n── LIME Analysis ──")

# Build LIME explainer on the transformed feature space
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=all_feature_names,
    class_names=["No Stroke", "Stroke"],
    mode="classification",
    random_state=SEED,
)

# Pick informative examples from the test set
stroke_mask = y_test == 1
no_stroke_mask = y_test == 0

# True positive: stroke patient correctly identified
tp_mask = (y_test == 1) & (y_pred_test == 1)
# False negative: stroke patient missed  
fn_mask = (y_test == 1) & (y_pred_test == 0)
# False positive: healthy patient flagged
fp_mask = (y_test == 0) & (y_pred_test == 1)
# True negative with high confidence
tn_mask = (y_test == 0) & (y_pred_test == 0)

cases = []
if tp_mask.any():
    idx = y_prob_test[tp_mask].argmax()  # highest confidence TP
    cases.append(("True Positive (highest conf)", tp_mask.values.nonzero()[0][idx]))
if fn_mask.any():
    idx = 0  # first FN
    cases.append(("False Negative (missed stroke)", fn_mask.values.nonzero()[0][idx]))
if fp_mask.any():
    idx = 0  # first FP
    cases.append(("False Positive (false alarm)", fp_mask.values.nonzero()[0][idx]))
if tn_mask.any():
    # pick a TN with the lowest probability (most confident no-stroke)
    tn_indices = tn_mask.values.nonzero()[0]
    idx = y_prob_test[tn_indices].argmin()
    cases.append(("True Negative (most confident)", tn_indices[idx]))

lime_records = []
for case_name, case_idx in cases:
    instance = X_test_transformed[case_idx]
    
    exp = lime_explainer.explain_instance(
        instance,
        fitted_model.predict_proba,
        num_features=10,
        labels=[1],
    )
    
    # Get the label key that LIME actually produced
    lime_label = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
    
    # Save LIME explanation as figure
    fig = exp.as_pyplot_figure(label=lime_label)
    fig.set_size_inches(10, 6)
    safe_name = case_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.title(f"LIME — {case_name}\n"
              f"P(Stroke) = {y_prob_test[case_idx]:.4f} | "
              f"True = {'Stroke' if y_test.iloc[case_idx] == 1 else 'No Stroke'}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"lime_{safe_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Collect explanation features
    for feat, weight in exp.as_list(label=lime_label):
        lime_records.append({
            "Case": case_name,
            "Index": case_idx,
            "P(Stroke)": y_prob_test[case_idx],
            "True Label": int(y_test.iloc[case_idx]),
            "Feature Condition": feat,
            "Weight": weight,
        })
    
    print(f"  ✅ {case_name} (idx={case_idx}, P(stroke)={y_prob_test[case_idx]:.4f})")

lime_df = pd.DataFrame(lime_records)
lime_df.to_csv(os.path.join(RES_DIR, "lime_explanations.csv"), index=False, float_format="%.4f")
print(f"  ✅ Saved: {RES_DIR}/lime_explanations.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 8. THRESHOLD ANALYSIS (from winner model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Threshold Sweep ──")

thresholds = np.arange(0.01, 0.99, 0.005)
thresh_rows = []
for t in thresholds:
    y_t = (y_prob_test >= t).astype(int)
    thresh_rows.append({
        "threshold": t,
        "F1":   f1_score(y_test, y_t, zero_division=0),
        "F2":   fbeta_score(y_test, y_t, beta=2, zero_division=0),
        "Recall": recall_score(y_test, y_t, zero_division=0),
        "Precision": precision_score(y_test, y_t, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_t),
    })
sw = pd.DataFrame(thresh_rows)

# Optimal thresholds
t_f1 = sw.loc[sw["F1"].idxmax(), "threshold"]
t_f2 = sw.loc[sw["F2"].idxmax(), "threshold"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sw["threshold"], sw["F1"],      color="#2563eb", lw=2.5, label="F1")
ax.plot(sw["threshold"], sw["F2"],      color="#7c3aed", lw=2,   label="F2")
ax.plot(sw["threshold"], sw["Recall"],  color="#16a34a", lw=2,   label="Recall")
ax.plot(sw["threshold"], sw["Precision"], color="#ea580c", lw=2, label="Precision")
ax.axvline(x=t_f1, color="#2563eb", ls="--", alpha=0.6, label=f"F1-opt t={t_f1:.3f}")
ax.axvline(x=t_f2, color="#7c3aed", ls=":",  alpha=0.6, label=f"F2-opt t={t_f2:.3f}")
ax.axvline(x=0.5,  color="gray",    ls=":",  alpha=0.4, label="default 0.5")
ax.set_xlim([0, 0.8])
ax.set_ylim([0, 1.05])
ax.set_xlabel("Decision Threshold", fontsize=12)
ax.set_ylabel("Metric Value", fontsize=12)
ax.set_title("Threshold Sweep — Winner Model", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "threshold_sweep.png"))
plt.close()

sw.to_csv(os.path.join(RES_DIR, "threshold_sweep.csv"), index=False, float_format="%.4f")
print(f"  F1-optimal threshold: {t_f1:.3f}")
print(f"  F2-optimal threshold: {t_f2:.3f}")
print(f"  ✅ Saved: {FIG_DIR}/threshold_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("WINNER ANALYSIS COMPLETE")
print("=" * 70)
print(f"\n📊 Figures → {FIG_DIR}/")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"    {f}")
print(f"\n📋 Tables  → {RES_DIR}/")
for f in sorted(os.listdir(RES_DIR)):
    print(f"    {f}")
print("=" * 70)
