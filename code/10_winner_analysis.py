"""
Winner Model Deep Analysis — XGBoost (Tuned)
CS 464 Stroke Prediction Project

Produces:
  1.  shap_beeswarm.png         — Global SHAP: feature impact distribution
  2.  shap_bar.png              — Global SHAP: mean |SHAP| ranking
  3.  shap_dependence_age.png   — Age vs SHAP, coloured by glucose
  4.  shap_dependence_glucose.png — Glucose vs SHAP, coloured by age
  5.  shap_waterfall_tp.png     — Local: true positive patient explanation
  6.  shap_waterfall_fn.png     — Local: false negative patient explanation
  7.  calibration_curve.png     — Reliability diagram before & after Platt scaling
  8.  roc_pr_curves.png         — ROC + PR curves with operating points marked
  9.  confusion_matrices.png    — At screening / balanced / confirmatory thresholds
  10. threshold_metrics.png     — F1, Recall, Precision vs threshold sweep
  11. metrics_radar.png         — Radar chart of all key metrics
  12. winner_metrics.csv        — Full metric table at all operating points
  13. mcc_analysis.csv          — MCC and additional metrics at each threshold
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.base import clone
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from shared import load_data, get_split, SEED, setup_plot_style, N_FOLDS

setup_plot_style()
plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 150})

OUT_DIR = os.path.join("outputs", "results", "winner_analysis")
FIG_DIR = os.path.join("outputs", "figures", "winner_analysis")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("WINNER MODEL: XGBoost (Tuned) — Deep Analysis")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# Data + Best Config
# ══════════════════════════════════════════════════════════════
df = load_data()
X_train, X_test, y_train, y_test = get_split(df)

NUMERIC     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type",
               "Residence_type", "smoking_status"]
PASSTHROUGH = ["hypertension", "heart_disease", "bmi_missing"]

# Best params from hyper_search
BEST_PARAMS = {
    "subsample":        1.0,
    "scale_pos_weight": 10,
    "n_estimators":     100,
    "max_depth":        5,
    "learning_rate":    0.01,
    "colsample_bytree": 0.7,
}

# Iterative imputer (best imputation strategy from tuning)
def get_iterative_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", ImbPipeline([
            ("imputer", IterativeImputer(random_state=SEED, max_iter=10)),
            ("scaler",  StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("pass", "passthrough", PASSTHROUGH),
    ])

preprocessor = get_iterative_preprocessor()
model = XGBClassifier(
    **BEST_PARAMS,
    eval_metric="logloss",
    random_state=SEED,
    verbosity=0,
)

# Full pipeline
pipe = ImbPipeline([
    ("preprocessor", clone(preprocessor)),
    ("model", model),
])

print("Training winner model...")
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Get feature names after preprocessing
prepped    = pipe.named_steps["preprocessor"]
num_names  = NUMERIC
cat_names  = prepped.named_transformers_["cat"].get_feature_names_out(CATEGORICAL).tolist()
feat_names = num_names + cat_names + PASSTHROUGH

# Preprocessed training data for SHAP
X_train_prep = prepped.transform(X_train)
X_test_prep  = prepped.transform(X_test)

print(f"AUC-PR={average_precision_score(y_test, y_prob):.4f}  "
      f"AUC-ROC={roc_auc_score(y_test, y_prob):.4f}  "
      f"Recall={recall_score(y_test, y_pred, zero_division=0):.4f}")

# ══════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════
print("\nComputing SHAP values (TreeSHAP — exact)...")
explainer   = shap.TreeExplainer(pipe.named_steps["model"])
shap_values = explainer.shap_values(X_test_prep)   # shape: (n_test, n_features)

shap_df = pd.DataFrame(shap_values, columns=feat_names)

# ── Plot 1: Beeswarm ─────────────────────────────────────────
print("Plot 1: SHAP beeswarm")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_prep,
                  feature_names=feat_names,
                  max_display=15,
                  show=False, plot_type="dot")
plt.title("SHAP Summary — Feature Impact on Stroke Prediction\n"
          "(XGBoost Tuned, TreeSHAP)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 2: Bar ──────────────────────────────────────────────
print("Plot 2: SHAP bar (mean |SHAP|)")
mean_shap = np.abs(shap_df).mean().sort_values(ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(mean_shap.index, mean_shap.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(mean_shap))))
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Global Feature Importance (Mean |SHAP|)\nXGBoost Tuned",
             fontsize=13, fontweight="bold")
for bar, val in zip(bars, mean_shap.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3 & 4: Dependence plots ────────────────────────────
for main_feat, color_feat in [("age", "avg_glucose_level"),
                               ("avg_glucose_level", "age")]:
    print(f"Plot: SHAP dependence {main_feat}")
    main_idx  = feat_names.index(main_feat)
    color_idx = feat_names.index(color_feat)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(X_test_prep[:, main_idx], shap_values[:, main_idx],
                    c=X_test_prep[:, color_idx],
                    cmap="coolwarm", alpha=0.6, s=30, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(color_feat.replace("_", " ").title(), fontsize=10)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel(main_feat.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(f"SHAP value for {main_feat.replace('_',' ').title()}", fontsize=11)
    ax.set_title(f"SHAP Dependence: {main_feat.replace('_',' ').title()}\n"
                 f"(colour = {color_feat.replace('_',' ').title()})",
                 fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"shap_dependence_{main_feat}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# ── Plot 5 & 6: Waterfall — TP and FN ───────────────────────
y_test_arr = y_test.values
tp_idx = np.where((y_test_arr == 1) & (y_pred == 1))[0]
fn_idx = np.where((y_test_arr == 1) & (y_pred == 0))[0]

for label, indices, fname in [
    ("True Positive (Stroke correctly detected)", tp_idx, "shap_waterfall_tp.png"),
    ("False Negative (Stroke missed by model)",   fn_idx, "shap_waterfall_fn.png"),
]:
    if len(indices) == 0:
        print(f"No {label} samples found, skipping.")
        continue
    idx = indices[0]
    print(f"Plot: Waterfall for {label}")
    exp = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_test_prep[idx],
        feature_names=feat_names,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(exp, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — {label}\nPredicted prob: {y_prob[idx]:.3f}",
              fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()

# ══════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════
print("\nCalibration analysis...")

# Platt scaling on training data
calibrated = CalibratedClassifierCV(pipe.named_steps["model"],
                                    method="sigmoid", cv="prefit")
calibrated.fit(X_train_prep, y_train)
y_prob_cal = calibrated.predict_proba(X_test_prep)[:, 1]

brier_raw = brier_score_loss(y_test, y_prob)
brier_cal = brier_score_loss(y_test, y_prob_cal)

print(f"Brier (raw):   {brier_raw:.4f}")
print(f"Brier (Platt): {brier_cal:.4f}  (Δ={brier_cal - brier_raw:+.4f})")

# Calibration curve plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, prob, title, brier in [
    (axes[0], y_prob,     f"Before Calibration\nBrier={brier_raw:.4f}", brier_raw),
    (axes[1], y_prob_cal, f"After Platt Scaling\nBrier={brier_cal:.4f}", brier_cal),
]:
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color="#2563eb", lw=2.5,
            markersize=7, label="XGBoost Tuned")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Perfect calibration")
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.1, color="#2563eb")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    sns.despine(ax=ax)

fig.suptitle("Reliability Diagram — XGBoost Tuned", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "calibration_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# ROC + PR curves with operating points
# ══════════════════════════════════════════════════════════════
print("ROC + PR curves...")

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
prec_c, rec_c, pr_thresholds = precision_recall_curve(y_test, y_prob)
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr  = average_precision_score(y_test, y_prob)

# Thresholds
thresholds = np.arange(0.01, 0.99, 0.005)
f1s, recs, precs = [], [], []
for t in thresholds:
    yp = (y_prob >= t).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
    recs.append(recall_score(y_test, yp, zero_division=0))
    precs.append(precision_score(y_test, yp, zero_division=0))
f1s = np.array(f1s); recs = np.array(recs); precs = np.array(precs)

t_balanced  = thresholds[np.argmax(f1s)]
t_screening = thresholds[np.where(recs >= 0.80)[0][-1]] if (recs >= 0.80).any() else thresholds[np.argmax(recs)]
t_confirm   = thresholds[np.where(precs >= 0.30)[0][0]]  if (precs >= 0.30).any() else thresholds[np.argmax(precs)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
ax = axes[0]
ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"XGBoost Tuned (AUC={auc_roc:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
for t_val, color, label in [
    (t_screening, "#ef4444", "Screening"),
    (t_balanced,  "#eab308", "Balanced"),
    (t_confirm,   "#22c55e", "Confirmatory"),
]:
    idx = np.argmin(np.abs(roc_thresholds - t_val))
    ax.scatter(fpr[idx], tpr[idx], color=color, s=120, zorder=5,
               label=f"{label} (t={t_val:.2f})", edgecolors="black", lw=0.8)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve",
       xlim=[0, 1], ylim=[0, 1.02])
ax.legend(fontsize=9); sns.despine(ax=ax)

# PR
ax = axes[1]
ax.plot(rec_c, prec_c, color="#ea580c", lw=2.5, label=f"XGBoost Tuned (AP={auc_pr:.3f})")
ax.axhline(y=y_test.mean(), color="k", ls="--", alpha=0.3, lw=1,
           label=f"Prevalence ({y_test.mean():.3f})")
for t_val, color, label in [
    (t_screening, "#ef4444", "Screening"),
    (t_balanced,  "#eab308", "Balanced"),
    (t_confirm,   "#22c55e", "Confirmatory"),
]:
    idx = np.argmin(np.abs(pr_thresholds - t_val))
    ax.scatter(rec_c[idx], prec_c[idx], color=color, s=120, zorder=5,
               label=f"{label} (t={t_val:.2f})", edgecolors="black", lw=0.8)
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve",
       xlim=[0, 1], ylim=[0, 1.02])
ax.legend(fontsize=9); sns.despine(ax=ax)

fig.suptitle("XGBoost Tuned — ROC and PR Curves with Clinical Operating Points",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Confusion matrices at 3 operating points
# ══════════════════════════════════════════════════════════════
print("Confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
op_configs = [
    (t_screening, "#ef4444", "Screening\n(High Recall)"),
    (t_balanced,  "#eab308", "Balanced\n(Max F1)"),
    (t_confirm,   "#22c55e", "Confirmatory\n(High Precision)"),
]
for ax, (t_val, color, label) in zip(axes, op_configs):
    yp = (y_prob >= t_val).astype(int)
    cm = confusion_matrix(y_test, yp)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{cm_norm[r,c]:.1f}%\n(n={cm[r,c]})"
                       for c in range(2)] for r in range(2)])
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, annot[r, c], ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm_norm[r, c] > 60 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Stroke", "Stroke"]); ax.set_yticklabels(["No Stroke", "Stroke"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    f1_t  = f1_score(y_test, yp, zero_division=0)
    rec_t = recall_score(y_test, yp, zero_division=0)
    prec_t = precision_score(y_test, yp, zero_division=0)
    ax.set_title(f"{label}\nt={t_val:.3f}  F1={f1_t:.3f}  Rec={rec_t:.3f}  Prec={prec_t:.3f}",
                 fontsize=10, fontweight="bold", color=color)

fig.suptitle("Confusion Matrices — XGBoost Tuned at Three Operating Points",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Threshold sweep: F1, Recall, Precision, MCC
# ══════════════════════════════════════════════════════════════
print("Threshold sweep + MCC...")
mccs = []
for t in thresholds:
    yp = (y_prob >= t).astype(int)
    mccs.append(matthews_corrcoef(y_test, yp))
mccs = np.array(mccs)

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(thresholds, f1s,   color="#2563eb", lw=2.5, label="F1")
ax.plot(thresholds, recs,  color="#22c55e", lw=2.0, label="Recall")
ax.plot(thresholds, precs, color="#ea580c", lw=2.0, label="Precision")
ax.plot(thresholds, mccs,  color="#a855f7", lw=2.0, ls="--", label="MCC")

for t_val, color, label in [
    (t_screening, "#ef4444", "Screening"),
    (t_balanced,  "#eab308", "Balanced"),
    (t_confirm,   "#22c55e", "Confirmatory"),
]:
    ax.axvline(x=t_val, color=color, ls=":", alpha=0.7, lw=1.5)
    ax.text(t_val + 0.005, 0.95, label, fontsize=8, color=color, va="top")

ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Metric Value", fontsize=11)
ax.set_title("Threshold Sweep — F1, Recall, Precision, MCC\nXGBoost Tuned",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="center right")
ax.set_xlim([0, 0.9]); ax.set_ylim([0, 1.05])
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "threshold_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Radar chart — all key metrics at balanced threshold
# ══════════════════════════════════════════════════════════════
print("Radar chart...")
yp_bal = (y_prob >= t_balanced).astype(int)
metrics_radar = {
    "AUC-PR":    auc_pr,
    "AUC-ROC":   auc_roc,
    "Recall":    recall_score(y_test, yp_bal, zero_division=0),
    "Precision": precision_score(y_test, yp_bal, zero_division=0),
    "F1":        f1_score(y_test, yp_bal, zero_division=0),
    "MCC":       max(0, matthews_corrcoef(y_test, yp_bal)),
    "Bal. Acc":  balanced_accuracy_score(y_test, yp_bal),
    "1-Brier":   1 - brier_raw,
}
labels   = list(metrics_radar.keys())
values   = list(metrics_radar.values())
N        = len(labels)
angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values  += values[:1]; angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, values, color="#2563eb", alpha=0.25)
ax.plot(angles, values, color="#2563eb", lw=2.5, marker="o", markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("XGBoost Tuned — Performance Radar\n(balanced threshold, all key metrics)",
             fontsize=13, fontweight="bold", pad=20)
for angle, value, label in zip(angles[:-1], values[:-1], labels):
    ax.annotate(f"{value:.3f}", xy=(angle, value),
                xytext=(angle, value + 0.07),
                ha="center", va="center", fontsize=9, color="#1e3a8a", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "metrics_radar.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Save CSVs
# ══════════════════════════════════════════════════════════════
print("\nSaving CSVs...")

# Full metrics at all 3 operating points
records = []
for t_val, op_name in [
    (t_screening, "Screening"),
    (t_balanced,  "Balanced (F1-optimal)"),
    (t_confirm,   "Confirmatory"),
    (0.5,         "Default (t=0.5)"),
]:
    yp = (y_prob >= t_val).astype(int)
    records.append({
        "Operating Point": op_name,
        "Threshold":       round(t_val, 4),
        "AUC-PR":          round(auc_pr, 4),
        "AUC-ROC":         round(auc_roc, 4),
        "F1":              round(f1_score(y_test, yp, zero_division=0), 4),
        "Recall":          round(recall_score(y_test, yp, zero_division=0), 4),
        "Precision":       round(precision_score(y_test, yp, zero_division=0), 4),
        "MCC":             round(matthews_corrcoef(y_test, yp), 4),
        "Balanced Acc":    round(balanced_accuracy_score(y_test, yp), 4),
        "Brier (raw)":     round(brier_raw, 4),
        "Brier (Platt)":   round(brier_cal, 4),
    })
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(os.path.join(OUT_DIR, "winner_metrics.csv"), index=False)
print(metrics_df.to_string(index=False))

# MCC + metrics across all thresholds
mcc_df = pd.DataFrame({
    "threshold": thresholds,
    "F1": f1s, "Recall": recs, "Precision": precs, "MCC": mccs,
})
mcc_df.to_csv(os.path.join(OUT_DIR, "mcc_analysis.csv"), index=False, float_format="%.4f")

# SHAP feature importance
shap_importance = pd.DataFrame({
    "feature":    feat_names,
    "mean_shap":  np.abs(shap_df).mean().values,
}).sort_values("mean_shap", ascending=False)
shap_importance.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False, float_format="%.4f")

print("\n" + "=" * 60)
print("WINNER ANALYSIS COMPLETE")
print(f"Figures  → {FIG_DIR}/")
print(f"Results  → {OUT_DIR}/")
print("=" * 60)
print("\nOutputs:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {FIG_DIR}/{f}")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {OUT_DIR}/{f}")
