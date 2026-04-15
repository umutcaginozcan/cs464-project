"""
Hyperparameter Tuning — Joint Optimisation of Imputation, Resampling & Model Params
CS 464 Stroke Prediction Project

Optimisation axes
─────────────────
1. BMI imputation strategy (4 options)
     median_noflag  SimpleImputer(median), no missing indicator
     median_flag    SimpleImputer(median) + bmi_missing binary column  ← current default
     knn            KNNImputer(k=5) + bmi_missing binary column
     iterative      IterativeImputer (MICE) + bmi_missing binary column

2. Resampling strategy (5 options)
     none           No resampling (class_weight handles imbalance)
     smote          Synthetic Minority Oversampling
     adasyn         Adaptive Synthetic Oversampling
     undersample    RandomUnderSampler
     smoteenn       SMOTE + Edited Nearest Neighbours (combined)

3. Model hyperparameters — via RandomizedSearchCV, one grid per model
     Logistic Regression, SVM (RBF), Random Forest, XGBoost

Primary metric: AUC-PR (Average Precision Score)
─────────────────────────────────────────────────
With a 4.87 % positive rate, AUC-ROC is inflated by the abundance of true
negatives — the model can score 0.85+ AUC-ROC while barely detecting strokes.
AUC-PR focuses entirely on the minority class (stroke = 1): it penalises both
missed detections (low recall) and false alarms (low precision), making it the
correct optimisation target for severely imbalanced medical datasets.

4. Threshold optimisation — post-hoc sweep on the best pipeline per model,
   reporting both the F1-optimal threshold and a high-recall screening threshold.

Outputs (outputs/results/hyper_search/)
  all_results.csv          every (imp, res, model) combination with metrics
  best_per_model.csv       winning config per model
  optimal_thresholds.csv   t* and screening threshold for each model
  heatmap_<model>.png      AUC-PR grid: imputation × resampling
  heatmap_model_resampling_<metric>.png  model × resampling across metrics
  imputation_comparison.png
  resampling_comparison.png
  best_per_model_bar.png
  threshold_sweep.png
"""

import sys
import os

# ── Path Setup ───────────────────────────────────────────────────────────────
_script_dir  = os.path.dirname(os.path.abspath(__file__))   # code/03_hyper/
_code_dir    = os.path.dirname(_script_dir)                  # code/
_project_dir = os.path.dirname(_code_dir)                   # project root

sys.path.insert(0, _code_dir)   # so we can `from shared import …`
os.chdir(_project_dir)          # so relative paths (data/, outputs/) work

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa – must come before IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score,
    precision_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier

from shared import load_data, get_split, SEED, N_FOLDS, setup_plot_style

warnings.filterwarnings("ignore")
setup_plot_style()

OUT_DIR = os.path.join("outputs", "results", "hyper_search")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column groups ────────────────────────────────────────────────────────────
_NUMERIC      = ["age", "avg_glucose_level", "bmi"]
_CATEGORICAL  = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
_PASS_FLAG    = ["hypertension", "heart_disease", "bmi_missing"]   # with indicator
_PASS_NOFLAG  = ["hypertension", "heart_disease"]                   # without indicator


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPUTATION STRATEGY FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def make_preprocessor(strategy: str) -> ColumnTransformer:
    """Return a fresh ColumnTransformer for the given BMI imputation strategy.

    All four options run imputation only on the numeric block (age,
    avg_glucose_level, bmi).  The bmi_missing flag is either included in
    passthrough (strategies with '_flag' or 'knn'/'iterative') or dropped
    (median_noflag), letting the model decide whether missingness itself is
    predictive.
    """
    _cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if strategy == "median_noflag":
        # Baseline imputation without signalling missingness to the model.
        num = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())])
        return ColumnTransformer([
            ("num",  num,         _NUMERIC),
            ("cat",  _cat,        _CATEGORICAL),
            ("pass", "passthrough", _PASS_NOFLAG),
        ])

    elif strategy == "median_flag":
        # Current project default: median + explicit missingness indicator.
        num = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())])
        return ColumnTransformer([
            ("num",  num,         _NUMERIC),
            ("cat",  _cat,        _CATEGORICAL),
            ("pass", "passthrough", _PASS_FLAG),
        ])

    elif strategy == "knn":
        # KNN imputation borrows information from age and glucose when
        # estimating BMI — more principled than a global median.
        num = ImbPipeline([("imp", KNNImputer(n_neighbors=5)),
                           ("sc",  StandardScaler())])
        return ColumnTransformer([
            ("num",  num,         _NUMERIC),
            ("cat",  _cat,        _CATEGORICAL),
            ("pass", "passthrough", _PASS_FLAG),
        ])

    elif strategy == "iterative":
        # MICE-style: iteratively regresses each feature on the others.
        # Most statistically principled but slowest.
        num = ImbPipeline([("imp", IterativeImputer(random_state=SEED, max_iter=10)),
                           ("sc",  StandardScaler())])
        return ColumnTransformer([
            ("num",  num,         _NUMERIC),
            ("cat",  _cat,        _CATEGORICAL),
            ("pass", "passthrough", _PASS_FLAG),
        ])

    raise ValueError(f"Unknown imputation strategy: {strategy!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESAMPLING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
RESAMPLERS = {
    "none":        None,
    "smote":       SMOTE(random_state=SEED),
    "adasyn":      ADASYN(random_state=SEED),
    "undersample": RandomUnderSampler(random_state=SEED),
    "smoteenn":    SMOTEENN(random_state=SEED),
}

IMPUTATION_STRATEGIES = ["median_noflag", "median_flag", "knn", "iterative"]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL HYPERPARAMETER GRIDS
# ══════════════════════════════════════════════════════════════════════════════
# All param keys use the "model__" prefix to match the pipeline step name.
# n_iter controls RandomizedSearchCV — kept high enough for reliable coverage
# but low enough for tractable runtime (SVM is the bottleneck).

MODELS_AND_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, solver="saga", random_state=SEED),
        "params": {
            "model__C":            [0.001, 0.01, 0.1, 1, 10, 100],
            "model__penalty":      ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
        "n_iter": 10,
    },
    "SVM (RBF)": {
        "model": SVC(kernel="rbf", probability=True, random_state=SEED),
        "params": {
            "model__C":            [0.1, 1, 10, 100],
            "model__gamma":        ["scale", "auto", 0.001, 0.01, 0.1],
            "model__class_weight": [None, "balanced"],
        },
        "n_iter": 10,
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=SEED),
        "params": {
            "model__n_estimators":     [100, 200, 300],
            "model__max_depth":        [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features":     ["sqrt", "log2"],
            "model__class_weight":     [None, "balanced", "balanced_subsample"],
        },
        "n_iter": 15,
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=SEED, eval_metric="logloss", verbosity=0,
        ),
        "params": {
            "model__n_estimators":     [100, 200, 300],
            "model__max_depth":        [3, 5, 7],
            "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "model__subsample":        [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__scale_pos_weight": [1, 5, 10, 20],
        },
        "n_iter": 20,
    },
}

CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN SEARCH LOOP
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
X_train, X_test, y_train, y_test = get_split(df)

total_jobs = len(IMPUTATION_STRATEGIES) * len(RESAMPLERS) * len(MODELS_AND_GRIDS)
print("=" * 70)
print("HYPERPARAMETER SEARCH — AUC-PR OPTIMISATION")
print(f"{len(IMPUTATION_STRATEGIES)} imputation × {len(RESAMPLERS)} resampling "
      f"× {len(MODELS_AND_GRIDS)} models = {total_jobs} search jobs")
print("=" * 70)

CHECKPOINT_PATH = os.path.join(OUT_DIR, "all_results.csv")

# Resume from checkpoint if it exists
if os.path.exists(CHECKPOINT_PATH):
    _ckpt = pd.read_csv(CHECKPOINT_PATH)
    all_results = _ckpt.to_dict("records")
    _done = {(r["imputation"], r["resampling"], r["model"]) for r in all_results}
    print(f"Resuming from checkpoint — {len(all_results)} jobs already done.")
else:
    all_results = []
    _done = set()

# Tracks the best pipeline found so far for each model (across all imp/res combos)
best_per_model: dict[str, dict] = {}

job = 0
for imp_name in IMPUTATION_STRATEGIES:
    for res_name, resampler in RESAMPLERS.items():
        for model_name, cfg in MODELS_AND_GRIDS.items():
            job += 1
            if (imp_name, res_name, model_name) in _done:
                print(f"[{job:02d}/{total_jobs}] SKIP (already done): {imp_name} | {res_name} | {model_name}")
                continue
            tag = f"[{job:02d}/{total_jobs}] {imp_name:<14} | {res_name:<11} | {model_name}"
            print(f"\n{tag}", flush=True)

            # ── Build imblearn pipeline ──────────────────────────────────────
            steps = [("preprocessor", make_preprocessor(imp_name))]
            if resampler is not None:
                steps.append(("resampler", clone(resampler)))
            steps.append(("model", clone(cfg["model"])))
            pipe = ImbPipeline(steps)

            # ── RandomizedSearchCV (scoring = AUC-PR) ────────────────────────
            # n_jobs=1: Windows'ta parallel spawn overhead'ı önler
            search = RandomizedSearchCV(
                pipe,
                param_distributions=cfg["params"],
                n_iter=cfg["n_iter"],
                scoring="average_precision",
                cv=CV,
                refit=True,
                n_jobs=1,
                random_state=SEED,
                error_score=0.0,
            )

            try:
                search.fit(X_train, y_train)
                best_pipe = search.best_estimator_
                cv_auc_pr = search.best_score_

                y_prob = best_pipe.predict_proba(X_test)[:, 1]
                y_pred = best_pipe.predict(X_test)

                auc_pr  = average_precision_score(y_test, y_prob)
                auc_roc = roc_auc_score(y_test, y_prob)
                f1      = f1_score(y_test, y_pred, zero_division=0)
                rec     = recall_score(y_test, y_pred, zero_division=0)
                prec    = precision_score(y_test, y_pred, zero_division=0)

                print(f"  CV AUC-PR={cv_auc_pr:.4f}  "
                      f"Test AUC-PR={auc_pr:.4f}  F1={f1:.4f}  Rec={rec:.4f}")

                all_results.append({
                    "imputation":   imp_name,
                    "resampling":   res_name,
                    "model":        model_name,
                    "cv_auc_pr":    cv_auc_pr,
                    "test_auc_pr":  auc_pr,
                    "test_auc_roc": auc_roc,
                    "test_f1":      f1,
                    "test_recall":  rec,
                    "test_precision": prec,
                    "best_params":  str(search.best_params_),
                })

                # Keep the best pipeline per model across all combos
                if (model_name not in best_per_model
                        or auc_pr > best_per_model[model_name]["auc_pr"]):
                    best_per_model[model_name] = {
                        "auc_pr":   auc_pr,
                        "pipeline": best_pipe,
                        "imp":      imp_name,
                        "res":      res_name,
                        "y_prob":   y_prob,
                    }

            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_results.append({
                    "imputation": imp_name, "resampling": res_name,
                    "model": model_name,
                    "cv_auc_pr": np.nan, "test_auc_pr": np.nan,
                    "test_auc_roc": np.nan, "test_f1": np.nan,
                    "test_recall": np.nan, "test_precision": np.nan,
                    "best_params": f"FAILED: {exc}",
                })

            # Checkpoint: her job sonrası kaydet — durursa veri kaybolmaz
            pd.DataFrame(all_results).to_csv(CHECKPOINT_PATH, index=False, float_format="%.4f")

# Persist full results table
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "all_results.csv"),
                  index=False, float_format="%.4f")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ANALYSIS & VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
valid = results_df.dropna(subset=["test_auc_pr"])

# ── 5a. Best combination per model ──────────────────────────────────────────
best_rows = (valid
             .loc[valid.groupby("model")["test_auc_pr"].idxmax()]
             .reset_index(drop=True))
best_rows.to_csv(os.path.join(OUT_DIR, "best_per_model.csv"),
                 index=False, float_format="%.4f")

print("\n" + "=" * 70)
print("BEST COMBINATION PER MODEL (test AUC-PR)")
print("=" * 70)
_cols = ["model", "imputation", "resampling",
         "test_auc_pr", "test_f1", "test_recall", "test_precision"]
print(best_rows[_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

global_best = valid.loc[valid["test_auc_pr"].idxmax()]
print(f"\nGlobal best → {global_best['model']} | "
      f"{global_best['imputation']} | {global_best['resampling']}")
print(f"  AUC-PR={global_best['test_auc_pr']:.4f}  "
      f"AUC-ROC={global_best['test_auc_roc']:.4f}  "
      f"F1={global_best['test_f1']:.4f}  "
      f"Recall={global_best['test_recall']:.4f}")

_COLORS = ["#2563eb", "#7c3aed", "#16a34a", "#ea580c"]

# ── 5b. Per-model heatmap: imputation × resampling ───────────────────────────
for model_name in MODELS_AND_GRIDS:
    sub = valid[valid["model"] == model_name]
    if sub.empty:
        continue
    pivot = sub.pivot_table(
        index="imputation", columns="resampling",
        values="test_auc_pr", aggfunc="max",
    )
    row_order = [r for r in IMPUTATION_STRATEGIES if r in pivot.index]
    col_order = [c for c in RESAMPLERS           if c in pivot.columns]
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "AUC-PR"})
    safe = (model_name.replace(" ", "_")
                      .replace("(", "").replace(")", ""))
    ax.set_title(f"{model_name} — AUC-PR: Imputation × Resampling",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Resampling Strategy")
    ax.set_ylabel("Imputation Strategy")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"heatmap_{safe}.png"))
    plt.close()

# ── 5c. Global heatmaps: model × resampling for AUC-PR / F1 / Recall ────────
for metric, label in [
    ("test_auc_pr",  "AUC-PR"),
    ("test_f1",      "F1"),
    ("test_recall",  "Recall"),
]:
    pivot = valid.pivot_table(
        index="model", columns="resampling",
        values=metric, aggfunc="max",
    )
    col_order = [c for c in RESAMPLERS if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": label})
    ax.set_title(
        f"Max {label} — Model × Resampling  (best imputation per cell)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"heatmap_model_resampling_{metric}.png"))
    plt.close()

# ── 5d. Imputation comparison (best resampling per model) ────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x_pos = np.arange(len(IMPUTATION_STRATEGIES))
width = 0.18
for i, model_name in enumerate(MODELS_AND_GRIDS):
    sub = valid[valid["model"] == model_name]
    vals = [
        sub.loc[sub["imputation"] == imp, "test_auc_pr"].max()
        if not sub[sub["imputation"] == imp].empty else 0.0
        for imp in IMPUTATION_STRATEGIES
    ]
    ax.bar(x_pos + (i - 1.5) * width, vals, width,
           label=model_name, color=_COLORS[i], alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(IMPUTATION_STRATEGIES, fontsize=10)
ax.set_ylabel("Test AUC-PR (best resampling)")
ax.set_title("Imputation Strategy Comparison (best resampling per model)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "imputation_comparison.png"))
plt.close()

# ── 5e. Resampling comparison (best imputation per model) ────────────────────
res_order = list(RESAMPLERS.keys())
fig, ax = plt.subplots(figsize=(13, 5))
x_pos2 = np.arange(len(res_order))
for i, model_name in enumerate(MODELS_AND_GRIDS):
    sub = valid[valid["model"] == model_name]
    vals = [
        sub.loc[sub["resampling"] == res, "test_auc_pr"].max()
        if not sub[sub["resampling"] == res].empty else 0.0
        for res in res_order
    ]
    ax.bar(x_pos2 + (i - 1.5) * width, vals, width,
           label=model_name, color=_COLORS[i], alpha=0.85)
ax.set_xticks(x_pos2)
ax.set_xticklabels(res_order, fontsize=10)
ax.set_ylabel("Test AUC-PR (best imputation)")
ax.set_title("Resampling Strategy Comparison (best imputation per model)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "resampling_comparison.png"))
plt.close()

# ── 5f. Best-per-model summary bar ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x_b = np.arange(len(best_rows))
bars = ax.bar(x_b, best_rows["test_auc_pr"].values,
              color=_COLORS[:len(best_rows)], width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
ax.set_xticks(x_b)
ax.set_xticklabels(
    [f"{r['model']}\n{r['imputation']}\n{r['resampling']}"
     for _, r in best_rows.iterrows()],
    fontsize=9,
)
ax.set_ylabel("Test AUC-PR")
ax.set_title("Optimal Configuration per Model", fontweight="bold", fontsize=13)
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "best_per_model_bar.png"))
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  THRESHOLD SWEEP ON BEST PIPELINE PER MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("THRESHOLD OPTIMISATION (best pipeline per model)")
print("=" * 70)

thresholds = np.arange(0.01, 0.99, 0.005)
threshold_records: list[dict] = []

n_models = len(best_per_model)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
axes = axes[0]

for ax, (model_name, info) in zip(axes, best_per_model.items()):
    y_prob = info["y_prob"]

    rows = []
    for t in thresholds:
        y_t = (y_prob >= t).astype(int)
        rows.append({
            "t":    t,
            "F1":   f1_score(y_test, y_t, zero_division=0),
            "Rec":  recall_score(y_test, y_t, zero_division=0),
            "Prec": precision_score(y_test, y_t, zero_division=0),
        })
    sw = pd.DataFrame(rows)

    # F1-optimal threshold
    idx_f1   = sw["F1"].idxmax()
    t_star   = sw.loc[idx_f1, "t"]
    f1_star  = sw.loc[idx_f1, "F1"]
    rec_star = sw.loc[idx_f1, "Rec"]
    pr_star  = sw.loc[idx_f1, "Prec"]

    # Screening threshold: highest t still achieving Recall >= 0.85
    screen_mask = sw["Rec"] >= 0.85
    t_screen = (sw.loc[screen_mask, "t"].iloc[-1]
                if screen_mask.any()
                else sw.loc[sw["Rec"].idxmax(), "t"])

    threshold_records.append({
        "model":               model_name,
        "imputation":          info["imp"],
        "resampling":          info["res"],
        "test_auc_pr":         info["auc_pr"],
        "t_star":              t_star,
        "F1_at_t_star":        f1_star,
        "Recall_at_t_star":    rec_star,
        "Precision_at_t_star": pr_star,
        "t_screen":            t_screen,
    })

    print(f"\n  {model_name}  [{info['imp']} + {info['res']}]")
    print(f"    Test AUC-PR  = {info['auc_pr']:.4f}")
    print(f"    F1-optimal   : t={t_star:.3f}  "
          f"F1={f1_star:.4f}  Rec={rec_star:.4f}  Prec={pr_star:.4f}")
    print(f"    Screening    : t={t_screen:.3f}  "
          f"(Recall >= 0.85)")

    ax.plot(sw["t"], sw["F1"],   color="#2563eb", lw=2.5, label="F1")
    ax.plot(sw["t"], sw["Rec"],  color="#16a34a", lw=2,   label="Recall")
    ax.plot(sw["t"], sw["Prec"], color="#ea580c", lw=2,   label="Precision")
    ax.axvline(x=t_star,   color="#eab308", ls="--", lw=1.5,
               label=f"F1-opt t={t_star:.2f}")
    ax.axvline(x=t_screen, color="#7c3aed", ls=":",  lw=1.2,
               label=f"Screen t={t_screen:.2f}")
    ax.axvline(x=0.5,      color="gray",    ls=":",  alpha=0.45,
               label="default 0.5")
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{model_name}\n{info['imp']} + {info['res']}", fontsize=10)
    ax.legend(fontsize=7)

sns.despine()
fig.suptitle("Threshold Sweep — Best Pipelines per Model",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "threshold_sweep.png"))
plt.close()

pd.DataFrame(threshold_records).to_csv(
    os.path.join(OUT_DIR, "optimal_thresholds.csv"),
    index=False, float_format="%.4f",
)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nGlobal Best Configuration:")
print(f"  Model       : {global_best['model']}")
print(f"  Imputation  : {global_best['imputation']}")
print(f"  Resampling  : {global_best['resampling']}")
print(f"  Test AUC-PR : {global_best['test_auc_pr']:.4f}")
print(f"  Test AUC-ROC: {global_best['test_auc_roc']:.4f}")
print(f"  Test F1     : {global_best['test_f1']:.4f}")
print(f"  Test Recall : {global_best['test_recall']:.4f}")
print(f"  Test Prec.  : {global_best['test_precision']:.4f}")
print(f"  Best Params : {global_best['best_params']}")
print(f"\nAll outputs saved to: {OUT_DIR}/")
print("=" * 70)
