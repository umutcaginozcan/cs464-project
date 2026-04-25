"""
SHAP Global Explainability + LIME + Permutation Importance
CS 464 Stroke Prediction Project

Produces:
  1. shap_beeswarm.png          — SHAP summary: feature impact distribution
  2. shap_bar.png               — SHAP: mean |SHAP| ranking
  3. permutation_importance.png — Permutation importance (AUC-PR drop)
  4. lime_global_bar.png        — Global LIME: mean |weight| across test set
  5. method_comparison.png      — 3-method side-by-side (normalized)
  6. shap_importance.csv
  7. permutation_importance.csv
  8. lime_importance.csv
  9. method_comparison.csv
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lime import lime_tabular

from shared import load_data, get_split, SEED, setup_plot_style, NUMERIC, CATEGORICAL, PASSTHROUGH

setup_plot_style()
plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 150})

OUT_DIR = os.path.join("outputs", "results", "shap_lime")
FIG_DIR = os.path.join("outputs", "figures", "shap_lime")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("SHAP + LIME + Permutation Importance — Global Explainability")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# Data + Winner Model
# ══════════════════════════════════════════════════════════════
df = load_data()
X_train, X_test, y_train, y_test = get_split(df)

BEST_PARAMS = {
    "subsample":        1.0,
    "scale_pos_weight": 10,
    "n_estimators":     100,
    "max_depth":        5,
    "learning_rate":    0.01,
    "colsample_bytree": 0.7,
}

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
model = XGBClassifier(**BEST_PARAMS, eval_metric="logloss", random_state=SEED, verbosity=0)
pipe = ImbPipeline([("preprocessor", clone(preprocessor)), ("model", model)])

print("Training XGBoost (Tuned)...")
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
print(f"AUC-PR = {average_precision_score(y_test, y_prob):.4f}")

prepped    = pipe.named_steps["preprocessor"]
cat_names  = prepped.named_transformers_["cat"].get_feature_names_out(CATEGORICAL).tolist()
feat_names = NUMERIC + cat_names + PASSTHROUGH

X_train_prep = prepped.transform(X_train)
X_test_prep  = prepped.transform(X_test)

# ══════════════════════════════════════════════════════════════
# 1. SHAP (TreeSHAP — exact)
# ══════════════════════════════════════════════════════════════
print("\n[1/3] Computing SHAP values (TreeSHAP — exact, full test set)...")
explainer   = shap.TreeExplainer(pipe.named_steps["model"])
shap_values = explainer.shap_values(X_test_prep)   # TreeSHAP hafıza dostu: O(n·d·L)
shap_df     = pd.DataFrame(shap_values, columns=feat_names)
mean_shap   = np.abs(shap_df).mean().sort_values(ascending=False)

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_prep, feature_names=feat_names,
                  max_display=15, show=False, plot_type="dot")
plt.title("SHAP Beeswarm — Feature Impact Distribution\n(XGBoost Tuned, TreeSHAP)",
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()

# Bar
top_shap = mean_shap.sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top_shap)))
bars = ax.barh(top_shap.index, top_shap.values, color=colors)
for bar, val in zip(bars, top_shap.values):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("Global Feature Importance — SHAP (Mean |SHAP|)\nXGBoost Tuned",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

shap_out = mean_shap.reset_index()
shap_out.columns = ["feature", "mean_abs_shap"]
shap_out["rank_shap"] = range(1, len(shap_out) + 1)
shap_out.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False)
print(f"  Top 5 SHAP: {list(mean_shap.head(5).index)}")

# ══════════════════════════════════════════════════════════════
# 2. Permutation Importance (AUC-PR scoring)
# ══════════════════════════════════════════════════════════════
print("\n[2/3] Computing Permutation Importance (n_repeats=10, AUC-PR, single-threaded)...")
perm = permutation_importance(pipe, X_test, y_test,
                              n_repeats=10, random_state=SEED,
                              scoring="average_precision", n_jobs=1)  # n_jobs=1: no process copies

perm_df = pd.DataFrame({
    "feature":  X_test.columns.tolist(),   # bmi_missing zaten X_test içinde
    "mean_imp": perm.importances_mean,
    "std_imp":  perm.importances_std,
}).sort_values("mean_imp", ascending=False).reset_index(drop=True)
perm_df["rank_perm"] = range(1, len(perm_df) + 1)
perm_df.to_csv(os.path.join(OUT_DIR, "permutation_importance.csv"), index=False)

top_perm = perm_df.sort_values("mean_imp").tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_perm)))
ax.barh(top_perm["feature"], top_perm["mean_imp"], color=colors,
        xerr=top_perm["std_imp"], capsize=3, ecolor="gray", error_kw={"alpha": 0.5})
ax.set_xlabel("Mean AUC-PR Drop (higher = more important)", fontsize=11)
ax.set_title("Permutation Importance (AUC-PR)\nXGBoost Tuned",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "permutation_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Top 5 Perm: {list(perm_df['feature'].head(5))}")

# ══════════════════════════════════════════════════════════════
# 3. LIME — Global approximation (mean |weight| over test set)
# ══════════════════════════════════════════════════════════════
LIME_SAMPLES = 30   # test örnekleri (az ama temsili)
LIME_PERTURB = 200  # her örnek için pertürbasyon sayısı (hafif)
print(f"\n[3/3] Computing LIME (global approx, {LIME_SAMPLES} samples × {LIME_PERTURB} perturbations)...")

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_prep,
    feature_names=feat_names,
    class_names=["No Stroke", "Stroke"],
    mode="classification",
    discretize_continuous=True,
    random_state=SEED,
)

np.random.seed(SEED)
sample_idx = np.random.choice(len(X_test_prep), size=LIME_SAMPLES, replace=False)
lime_weights = {f: [] for f in feat_names}

for i, idx in enumerate(sample_idx):
    if i % 10 == 0:
        print(f"  LIME sample {i}/{LIME_SAMPLES}...")
    exp = lime_explainer.explain_instance(
        X_test_prep[idx],
        lambda x: pipe.named_steps["model"].predict_proba(x),
        num_features=len(feat_names),
        num_samples=LIME_PERTURB,
    )
    exp_map = dict(exp.as_list())
    for feat in feat_names:
        matched = next((v for k, v in exp_map.items() if feat in k), 0.0)
        lime_weights[feat].append(abs(matched))

lime_df = pd.DataFrame({
    "feature":       feat_names,
    "mean_abs_lime": [np.mean(lime_weights[f]) for f in feat_names],
}).sort_values("mean_abs_lime", ascending=False).reset_index(drop=True)
lime_df["rank_lime"] = range(1, len(lime_df) + 1)
lime_df.to_csv(os.path.join(OUT_DIR, "lime_importance.csv"), index=False)

top_lime = lime_df.sort_values("mean_abs_lime").tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(top_lime)))
ax.barh(top_lime["feature"], top_lime["mean_abs_lime"], color=colors)
ax.set_xlabel("Mean |LIME weight| (100 test samples)", fontsize=11)
ax.set_title("Global LIME Feature Importance\nXGBoost Tuned",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lime_global_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Top 5 LIME: {list(lime_df['feature'].head(5))}")

# ══════════════════════════════════════════════════════════════
# 4. Three-Method Comparison (normalized 0-1)
# ══════════════════════════════════════════════════════════════
print("\nBuilding 3-method comparison figure...")

# Normalize each method to [0, 1]
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

shap_norm = norm01(mean_shap)
perm_norm  = norm01(perm_df.set_index("feature")["mean_imp"].reindex(feat_names).fillna(0))
lime_norm  = norm01(lime_df.set_index("feature")["mean_abs_lime"].reindex(feat_names).fillna(0))

# Keep top 12 by SHAP for the comparison
top12 = mean_shap.head(12).index.tolist()

comp_df = pd.DataFrame({
    "feature":    top12,
    "SHAP":       shap_norm[top12].values,
    "Permutation":perm_norm[top12].values,
    "LIME":       lime_norm[top12].values,
})
comp_df.to_csv(os.path.join(OUT_DIR, "method_comparison.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
method_styles = [
    ("SHAP",        comp_df["SHAP"],        plt.cm.RdYlGn_r,  "Mean |SHAP| (normalized)"),
    ("Permutation", comp_df["Permutation"], plt.cm.Blues,      "AUC-PR Drop (normalized)"),
    ("LIME",        comp_df["LIME"],        plt.cm.Oranges,    "Mean |LIME weight| (normalized)"),
]

# Clinical alignment flag
clinical_top = {"age", "avg_glucose_level"}
top3_shap = set(mean_shap.head(3).index)
clinical_match = clinical_top.issubset(top3_shap)

for ax, (method, values, cmap, xlabel) in zip(axes, method_styles):
    sorted_idx = values.argsort()
    feats_sorted = [comp_df["feature"].iloc[i] for i in sorted_idx]
    vals_sorted  = values.iloc[sorted_idx].values
    colors = cmap(np.linspace(0.25, 0.9, len(feats_sorted)))
    bars = ax.barh(feats_sorted, vals_sorted, color=colors)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(method, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)
    for bar, val in zip(bars, vals_sorted):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7.5)

clinical_label = ("✓ Age & Glucose in SHAP top-3 — matches clinical expectation"
                  if clinical_match else
                  "⚠ Age/Glucose NOT in SHAP top-3 — check feature names")
fig.suptitle(
    f"Feature Importance: Three-Method Comparison (Top 12 by SHAP)\n{clinical_label}",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "method_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SHAP + LIME + Permutation Importance — COMPLETE")
print("=" * 60)
print(f"\nTop 5 features per method:")
print(f"  SHAP:        {list(mean_shap.head(5).index)}")
print(f"  Permutation: {list(perm_df['feature'].head(5))}")
print(f"  LIME:        {list(lime_df['feature'].head(5))}")
print(f"\nClinical alignment (age + glucose in SHAP top-3): {clinical_match}")
print(f"\nFigures → {FIG_DIR}/")
print(f"Results → {OUT_DIR}/")
