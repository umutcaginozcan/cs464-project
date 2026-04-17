"""
09_calibration_summary.py
-----------------------------------------------------------------------------
Calibration Analysis + Master Experiment Summary Table
CS 464 Stroke Prediction Project

What this script does
----------------------
1. Trains every key model from every experiment on the same train/test split
2. Splits X_train → X_fit (80%) + X_calib (20%) for calibration fitting
3. Applies Platt Scaling (sigmoid) and Isotonic Regression to each model
4. Plots reliability diagrams ("if model says 70%, is it really 70%?")
5. Computes and compares Brier scores: raw vs Platt vs Isotonic
6. Trains the three MLP variants (BCE / Weighted / Focal) inline
7. Produces a master summary CSV + multi-panel summary figure covering
   ALL experiments: Exp-0 → Exp-1 → Exp-2 → Exp-3 → Hyper-Tuned → DL
"""

import sys
import os
import warnings
import traceback

# -- Path setup ---------------------------------------------------------------
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
os.chdir(_project_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer          # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

from shared import (
    load_data, get_split, get_preprocessor,
    SEED, NUMERIC, CATEGORICAL, PASSTHROUGH, setup_plot_style,
)

warnings.filterwarnings("ignore")
setup_plot_style()

OUT_DIR = os.path.join("outputs", "results", "calibration_summary")
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 72)
print("CALIBRATION ANALYSIS + MASTER SUMMARY TABLE")
print("=" * 72)

# ===========================================================================
# 1.  DATA
# ===========================================================================
df = load_data()
X_train_full, X_test, y_train_full, y_test = get_split(df)

# Calibration split: 80 % model fitting, 20 % calibrator fitting
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train_full, y_train_full,
    test_size=0.20, stratify=y_train_full, random_state=SEED,
)

n_pos  = int(y_train_full.sum())
n_neg  = int(len(y_train_full) - n_pos)
SPW    = n_neg / n_pos          # XGBoost scale_pos_weight

print(f"  X_fit   : {len(X_fit):5d}  (stroke={int(y_fit.sum())})")
print(f"  X_calib : {len(X_calib):5d}  (stroke={int(y_calib.sum())})")
print(f"  X_test  : {len(X_test):5d}  (stroke={int(y_test.sum())})")
print(f"  Prevalence test: {y_test.mean():.4f}")
print(f"  scale_pos_weight: {SPW:.2f}")

NAIVE_BRIER = brier_score_loss(y_test, np.full(len(y_test), y_test.mean()))
print(f"  Naive Brier baseline: {NAIVE_BRIER:.4f}\n")

# ===========================================================================
# 2.  PREPROCESSOR FACTORIES
# ===========================================================================
_PASS_FLAG   = ["hypertension", "heart_disease", "bmi_missing"]
_PASS_NOFLAG = ["hypertension", "heart_disease"]

def _cat():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

def pre_median_noflag():
    num = ImbPipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc",  StandardScaler())])
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", _cat(), CATEGORICAL),
                              ("pass", "passthrough", _PASS_NOFLAG)])

def pre_median_flag():          # same as shared.get_preprocessor()
    return get_preprocessor()

def pre_knn():
    num = ImbPipeline([("imp", KNNImputer(n_neighbors=5)),
                       ("sc",  StandardScaler())])
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", _cat(), CATEGORICAL),
                              ("pass", "passthrough", _PASS_FLAG)])

def pre_iterative():
    num = ImbPipeline([("imp", IterativeImputer(random_state=SEED, max_iter=10)),
                       ("sc",  StandardScaler())])
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", _cat(), CATEGORICAL),
                              ("pass", "passthrough", _PASS_FLAG)])

# ===========================================================================
# 3.  PIPELINE BUILDER
# ===========================================================================
def pipe(model, resampler=None, preprocessor_fn=None):
    """Build an ImbPipeline: preprocessor [→ resampler] → model."""
    prep = (preprocessor_fn() if preprocessor_fn is not None
            else get_preprocessor())
    steps = [("preprocessor", prep)]
    if resampler is not None:
        steps.append(("resampler", resampler))
    steps.append(("model", model))
    return ImbPipeline(steps)

# ===========================================================================
# 4.  EXPERIMENT CONFIGURATIONS
#     Each entry: (display_name, experiment_group, pipeline_object)
# ===========================================================================
CONFIGS = [
    # -- Exp-0: Baseline (no imbalance handling) --------------------------
    ("LR — Baseline",   "Exp-0 Baseline",
     pipe(LogisticRegression(max_iter=1000, random_state=SEED))),

    ("GNB — Baseline",  "Exp-0 Baseline",
     pipe(GaussianNB())),

    ("KNN-5 — Baseline","Exp-0 Baseline",
     pipe(KNeighborsClassifier(n_neighbors=5))),

    ("SVM — Baseline",  "Exp-0 Baseline",
     pipe(SVC(probability=True, random_state=SEED))),

    ("RF — Baseline",   "Exp-0 Baseline",
     pipe(RandomForestClassifier(n_estimators=200, random_state=SEED))),

    ("XGB — Baseline",  "Exp-0 Baseline",
     pipe(XGBClassifier(n_estimators=200, random_state=SEED,
                        eval_metric="logloss", verbosity=0))),

    # -- Exp-1: Class Weighting -------------------------------------------
    ("LR — ClassWeight","Exp-1 ClassWeight",
     pipe(LogisticRegression(class_weight="balanced",
                             max_iter=1000, random_state=SEED))),

    ("SVM — ClassWeight","Exp-1 ClassWeight",
     pipe(SVC(class_weight="balanced", probability=True, random_state=SEED))),

    ("RF — ClassWeight","Exp-1 ClassWeight",
     pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=SEED))),

    ("XGB — ClassWeight","Exp-1 ClassWeight",
     pipe(XGBClassifier(n_estimators=200, scale_pos_weight=SPW,
                        random_state=SEED, eval_metric="logloss", verbosity=0))),

    # -- Exp-2: SMOTE ----------------------------------------------------
    ("LR — SMOTE",      "Exp-2 SMOTE",
     pipe(LogisticRegression(max_iter=1000, random_state=SEED),
          SMOTE(random_state=SEED))),

    ("GNB — SMOTE",     "Exp-2 SMOTE",
     pipe(GaussianNB(), SMOTE(random_state=SEED))),

    ("SVM — SMOTE",     "Exp-2 SMOTE",
     pipe(SVC(probability=True, random_state=SEED),
          SMOTE(random_state=SEED))),

    ("RF — SMOTE",      "Exp-2 SMOTE",
     pipe(RandomForestClassifier(n_estimators=200, random_state=SEED),
          SMOTE(random_state=SEED))),

    ("XGB — SMOTE",     "Exp-2 SMOTE",
     pipe(XGBClassifier(n_estimators=200, random_state=SEED,
                        eval_metric="logloss", verbosity=0),
          SMOTE(random_state=SEED))),

    # -- Exp-3: ADASYN ---------------------------------------------------
    ("LR — ADASYN",     "Exp-3 ADASYN",
     pipe(LogisticRegression(max_iter=1000, random_state=SEED),
          ADASYN(random_state=SEED))),

    ("SVM — ADASYN",    "Exp-3 ADASYN",
     pipe(SVC(probability=True, random_state=SEED),
          ADASYN(random_state=SEED))),

    ("XGB — ADASYN",    "Exp-3 ADASYN",
     pipe(XGBClassifier(n_estimators=200, random_state=SEED,
                        eval_metric="logloss", verbosity=0),
          ADASYN(random_state=SEED))),

    # -- Hyper-Tuned (exact best params from hyper_search/best_per_model.csv) -
    # LR: median_noflag + undersample, C=1, l2, class_weight=None
    ("LR — Tuned",      "Hyper Tuned",
     pipe(LogisticRegression(C=1, penalty="l2", class_weight=None,
                             max_iter=2000, solver="saga", random_state=SEED),
          RandomUnderSampler(random_state=SEED),
          pre_median_noflag)),

    # SVM: knn + smote, C=100, gamma=0.001, class_weight=balanced
    ("SVM — Tuned",     "Hyper Tuned",
     pipe(SVC(C=100, gamma=0.001, class_weight="balanced",
              probability=True, random_state=SEED),
          SMOTE(random_state=SEED),
          pre_knn)),

    # RF: knn + undersample, n_est=200, min_leaf=5, max_feat=sqrt, depth=20, cw=balanced_subsample
    ("RF — Tuned",      "Hyper Tuned",
     pipe(RandomForestClassifier(n_estimators=200, max_depth=20,
                                 min_samples_leaf=5, max_features="sqrt",
                                 class_weight="balanced_subsample",
                                 random_state=SEED),
          RandomUnderSampler(random_state=SEED),
          pre_knn)),

    # XGB: iterative + none, depth=5, lr=0.01, n_est=100, sub=1.0, col=0.7, spw=10
    ("XGB — Tuned",     "Hyper Tuned",
     pipe(XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01,
                        subsample=1.0, colsample_bytree=0.7,
                        scale_pos_weight=10,
                        random_state=SEED, eval_metric="logloss", verbosity=0),
          None,
          pre_iterative)),
]

# ===========================================================================
# 5.  HELPERS: metrics + threshold sweep
# ===========================================================================
_THRESHOLDS = np.arange(0.01, 0.99, 0.005)

def compute_metrics(y_true, y_prob, y_pred=None):
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
        "AUC-PR":    average_precision_score(y_true, y_prob),
        "Brier":     brier_score_loss(y_true, y_prob),
    }

def threshold_optimise(y_true, y_prob):
    """Return (best_t, F1_at_t, Recall_at_t, Precision_at_t)."""
    best_f1, best_t, best_rec, best_prec = 0.0, 0.5, 0.0, 0.0
    for t in _THRESHOLDS:
        yt = (y_prob >= t).astype(int)
        ft = f1_score(y_true, yt, zero_division=0)
        if ft > best_f1:
            best_f1  = ft
            best_t   = t
            best_rec = recall_score(y_true, yt, zero_division=0)
            best_prec = precision_score(y_true, yt, zero_division=0)
    return best_t, best_f1, best_rec, best_prec

def manual_platt(y_calib_prob, y_calib_true, y_test_prob):
    """Fit logistic regression on calibration probs → transform test probs."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
    lr.fit(y_calib_prob.reshape(-1, 1), y_calib_true)
    return lr.predict_proba(y_test_prob.reshape(-1, 1))[:, 1]

def manual_isotonic(y_calib_prob, y_calib_true, y_test_prob):
    """Fit isotonic regression on calibration probs → transform test probs."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_calib_prob, y_calib_true)
    return iso.predict(y_test_prob)

# ===========================================================================
# 6.  TRAIN ALL SKLEARN MODELS
# ===========================================================================
print("-" * 72)
print("TRAINING SKLEARN MODELS")
print("-" * 72)

all_results   = []   # final summary rows
calib_store   = {}   # name → {raw, platt, isotonic} probs on X_test

for name, exp_label, pipeline in CONFIGS:
    print(f"  [{exp_label:18s}]  {name} ...", end=" ", flush=True)
    try:
        # -- Full train: for final metrics (same data as all experiments) --
        p_full = clone(pipeline)
        p_full.fit(X_train_full, y_train_full)
        y_prob_full = p_full.predict_proba(X_test)[:, 1]
        y_pred_full = p_full.predict(X_test)

        mets = compute_metrics(y_test.values, y_prob_full, y_pred_full)
        t_star, f1_t, rec_t, prec_t = threshold_optimise(y_test.values, y_prob_full)

        # -- Calibration: fit on X_fit, calibrate on X_calib -------------
        p_fit = clone(pipeline)
        p_fit.fit(X_fit, y_fit)
        y_calib_raw  = p_fit.predict_proba(X_calib)[:, 1]
        y_test_raw_c = p_fit.predict_proba(X_test)[:, 1]   # same model, test preds

        y_test_platt = manual_platt(y_calib_raw, y_calib.values, y_test_raw_c)
        y_test_iso   = manual_isotonic(y_calib_raw, y_calib.values, y_test_raw_c)

        calib_store[name] = {
            "raw":      y_test_raw_c,
            "platt":    y_test_platt,
            "isotonic": y_test_iso,
        }

        brier_platt = brier_score_loss(y_test.values, y_test_platt)
        brier_iso   = brier_score_loss(y_test.values, y_test_iso)

        all_results.append({
            "Model":            name,
            "Experiment":       exp_label,
            **mets,
            "Brier (Platt)":    brier_platt,
            "Brier (Isotonic)": brier_iso,
            "t*":               round(t_star, 3),
            "F1 @ t*":          f1_t,
            "Recall @ t*":      rec_t,
            "Precision @ t*":   prec_t,
        })
        print(f"AUC-PR={mets['AUC-PR']:.4f}  "
              f"Brier={mets['Brier']:.4f}→{min(brier_platt,brier_iso):.4f}")

    except Exception as exc:
        print(f"FAILED: {exc}")
        traceback.print_exc()
        all_results.append({
            "Model": name, "Experiment": exp_label,
            **{k: np.nan for k in ["Accuracy","Precision","Recall","F1",
                                    "AUC-ROC","AUC-PR","Brier",
                                    "Brier (Platt)","Brier (Isotonic)",
                                    "t*","F1 @ t*","Recall @ t*","Precision @ t*"]},
        })

# ===========================================================================
# 7.  PYTORCH MLP — train inline
# ===========================================================================
print("\n-" * 36)
print("DEEP LEARNING — PyTorch MLP (inline training)")
print("-" * 72)

_DL_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")

    # -- Preprocess once ---------------------------------------------------
    # Use the same shared preprocessor.  MLP shares the 80/20 train/test split.
    # Additionally create a val split from X_fit for early stopping,
    # and use X_calib as the calibration set (matches sklearn section).
    from shared import get_preprocessor as _gpp
    _prep_dl = _gpp()

    X_fit_val, X_val_dl, y_fit_val, y_val_dl = train_test_split(
        X_fit, y_fit, test_size=0.20, stratify=y_fit, random_state=SEED,
    )

    X_fit_val_t = torch.tensor(
        _prep_dl.fit_transform(X_fit_val).astype(np.float32))
    X_val_dl_t  = torch.tensor(
        _prep_dl.transform(X_val_dl).astype(np.float32))
    X_calib_t   = torch.tensor(
        _prep_dl.transform(X_calib).astype(np.float32))
    X_test_t    = torch.tensor(
        _prep_dl.transform(X_test).astype(np.float32))

    y_fit_val_t = torch.tensor(y_fit_val.values.astype(np.float32)).view(-1, 1)
    y_val_dl_t  = torch.tensor(y_val_dl.values.astype(np.float32)).view(-1, 1)
    y_calib_t   = torch.tensor(y_calib.values.astype(np.float32)).view(-1, 1)
    y_test_t    = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

    INPUT_DIM = X_fit_val_t.shape[1]

    # pos_weight from the inner fit set
    _n_pos_dl = int(y_fit_val_t.sum().item())
    _n_neg_dl = len(y_fit_val_t) - _n_pos_dl
    POS_WEIGHT = torch.tensor([_n_neg_dl / _n_pos_dl], dtype=torch.float32)

    def make_loader(X, y, shuffle=False, bs=64):
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    train_loader = make_loader(X_fit_val_t, y_fit_val_t, shuffle=True)
    val_loader   = make_loader(X_val_dl_t,  y_val_dl_t)

    class StrokeMLP(nn.Module):
        def __init__(self, inp, h1=64, h2=32, drop=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(inp, h1), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h1, h2),  nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h2, 1),
            )
        def forward(self, x):
            return self.net(x)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        def forward(self, inp, tgt):
            bce = nn.functional.binary_cross_entropy_with_logits(
                inp, tgt, reduction="none")
            pt  = torch.exp(-bce)
            return (((1 - pt) ** self.gamma) * bce).mean()

    def train_mlp(criterion, tag):
        model = StrokeMLP(INPUT_DIM).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        best_auc, patience, no_imp = -1.0, 10, 0
        best_state = None
        for epoch in range(100):
            model.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(X_val_dl_t)
                probs  = torch.sigmoid(logits).numpy().flatten()
            auc = average_precision_score(y_val_dl.values, probs)
            if auc > best_auc:
                best_auc = auc; no_imp = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        print(f"      [{tag}] stopped at epoch {epoch+1}, best val AUC-PR={best_auc:.4f}")
        return model

    MLP_VARIANTS = [
        ("MLP (BCE)",      nn.BCEWithLogitsLoss(),              "Deep Learning"),
        ("MLP (Weighted)", nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT), "Deep Learning"),
        ("MLP (Focal)",    FocalLoss(gamma=2.0),                "Deep Learning"),
    ]

    for mlp_name, criterion, exp_label in MLP_VARIANTS:
        print(f"  {mlp_name} ...", flush=True)
        try:
            model = train_mlp(criterion, mlp_name)

            def _probs(X_t):
                with torch.no_grad():
                    return torch.sigmoid(model(X_t)).numpy().flatten()

            y_calib_mlp  = _probs(X_calib_t)
            y_test_mlp   = _probs(X_test_t)
            y_pred_mlp   = (y_test_mlp >= 0.5).astype(int)

            mets = compute_metrics(y_test.values, y_test_mlp, y_pred_mlp)
            t_star, f1_t, rec_t, prec_t = threshold_optimise(
                y_test.values, y_test_mlp)

            y_test_platt = manual_platt(y_calib_mlp, y_calib.values, y_test_mlp)
            y_test_iso   = manual_isotonic(y_calib_mlp, y_calib.values, y_test_mlp)

            calib_store[mlp_name] = {
                "raw":      y_test_mlp,
                "platt":    y_test_platt,
                "isotonic": y_test_iso,
            }

            brier_platt = brier_score_loss(y_test.values, y_test_platt)
            brier_iso   = brier_score_loss(y_test.values, y_test_iso)

            all_results.append({
                "Model":            mlp_name,
                "Experiment":       exp_label,
                **mets,
                "Brier (Platt)":    brier_platt,
                "Brier (Isotonic)": brier_iso,
                "t*":               round(t_star, 3),
                "F1 @ t*":          f1_t,
                "Recall @ t*":      rec_t,
                "Precision @ t*":   prec_t,
            })
            print(f"      AUC-PR={mets['AUC-PR']:.4f}  "
                  f"Brier={mets['Brier']:.4f}→{min(brier_platt,brier_iso):.4f}")
            _DL_AVAILABLE = True

        except Exception as exc:
            print(f"      FAILED: {exc}")
            traceback.print_exc()

except ImportError as ie:
    print(f"  PyTorch not available: {ie}")

# ── Fallback: read pre-computed DL results from CSV ──────────────────────────
_DL_CSV = os.path.join("outputs", "results", "dl_mlp", "test_results.csv")
if not _DL_AVAILABLE and os.path.exists(_DL_CSV):
    print(f"  Loading pre-computed DL results from {_DL_CSV}")
    _dl_df = pd.read_csv(_DL_CSV)
    # Rename columns to match our schema
    _col_map = {"Model": "Model", "Accuracy": "Accuracy",
                "F1": "F1", "Recall": "Recall", "Precision": "Precision",
                "AUC-ROC": "AUC-ROC", "AUC-PR": "AUC-PR"}
    for _, row in _dl_df.iterrows():
        mlp_name = row["Model"]
        # Rename to match display convention
        mlp_name_clean = (mlp_name.replace("PyTorch MLP", "MLP")
                                  .replace("(Weighted)", "(Weighted BCE)"))
        # Brier cannot be computed without raw probabilities
        all_results.append({
            "Model":            mlp_name_clean,
            "Experiment":       "Deep Learning",
            "Accuracy":         row.get("Accuracy",  np.nan),
            "Precision":        row.get("Precision", np.nan),
            "Recall":           row.get("Recall",    np.nan),
            "F1":               row.get("F1",        np.nan),
            "AUC-ROC":          row.get("AUC-ROC",   np.nan),
            "AUC-PR":           row.get("AUC-PR",    np.nan),
            "Brier":            np.nan,   # probabilities not available
            "Brier (Platt)":    np.nan,
            "Brier (Isotonic)": np.nan,
            "t*":               np.nan,
            "F1 @ t*":          np.nan,
            "Recall @ t*":      np.nan,
            "Precision @ t*":   np.nan,
        })
        print(f"    {mlp_name_clean}: AUC-PR={row.get('AUC-PR', 'N/A'):.4f}")

# ===========================================================================
# 8.  MASTER SUMMARY TABLE
# ===========================================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "master_summary.csv"),
                  index=False, float_format="%.4f")

print("\n" + "=" * 72)
print("MASTER SUMMARY TABLE (sorted by AUC-PR)")
print("=" * 72)
_DISP = ["Model", "Experiment", "Recall", "Precision", "F1",
         "AUC-ROC", "AUC-PR", "Brier", "Brier (Platt)", "Brier (Isotonic)",
         "F1 @ t*", "t*"]
_sorted = results_df.sort_values("AUC-PR", ascending=False).reset_index(drop=True)
print(_sorted[_DISP].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ===========================================================================
# 9.  RELIABILITY DIAGRAMS
# ===========================================================================
print("\n--- Reliability Diagrams ---")

# Key models to visualise — one per experiment group + tuned + MLP
RELIABILITY_KEYS = [
    "LR — Baseline",
    "SVM — Baseline",
    "LR — ClassWeight",
    "LR — SMOTE",
    "SVM — SMOTE",
    "XGB — SMOTE",
    "LR — Tuned",
    "SVM — Tuned",
]
# Add MLP models if available
for _m in ["MLP (BCE)", "MLP (Weighted)", "MLP (Focal)"]:
    if _m in calib_store:
        RELIABILITY_KEYS.append(_m)

RELIABILITY_KEYS = [k for k in RELIABILITY_KEYS if k in calib_store]
N_REL = len(RELIABILITY_KEYS)
N_COLS = 3
N_ROWS = (N_REL + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(6.5 * N_COLS, 5.5 * N_ROWS))
axes = np.atleast_2d(axes).flatten()

for i, model_name in enumerate(RELIABILITY_KEYS):
    ax   = axes[i]
    stor = calib_store[model_name]
    y_true = y_test.values

    # -- Reliability curves ------------------------------------------------
    VARIANTS = [
        ("Uncalibrated",    "#2563eb", "o-",  stor["raw"]),
        ("Platt (Sigmoid)", "#ea580c", "s--", stor["platt"]),
        ("Isotonic Reg.",   "#16a34a", "^:",  stor["isotonic"]),
    ]
    for label, color, lstyle, probs in VARIANTS:
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, probs, n_bins=10, strategy="quantile")
            bs = brier_score_loss(y_true, probs)
            ax.plot(mean_pred, frac_pos, lstyle, color=color,
                    linewidth=2, markersize=5,
                    label=f"{label}  BS={bs:.4f}")
        except Exception:
            pass

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1.5,
            label="Perfect")

    # Prevalence reference
    prev = y_true.mean()
    ax.axhline(prev, color="#888", linestyle=":", linewidth=1.0, alpha=0.6,
               label=f"Prevalence {prev:.3f}")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability", fontsize=9)
    ax.set_ylabel("Fraction of Positives", fontsize=9)
    ax.set_title(model_name, fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # -- Probability histogram (twin axis, bottom-anchored) ----------------
    ax2 = ax.twinx()
    ax2.hist(stor["raw"], bins=20, color="#2563eb", alpha=0.12,
             density=True, label="P(y=1) dist.")
    ax2.set_ylabel("Density", fontsize=7, color="#666")
    ax2.tick_params(axis="y", labelcolor="#666", labelsize=6)
    # Push histogram to lower portion of plot
    ax2.set_ylim([0, ax2.get_ylim()[1] * 5])
    ax2.set_yticks([])

for j in range(N_REL, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Reliability Diagrams  —  Is the model's confidence justified?\n"
    "(Quantile binning, 10 bins; closer to diagonal = better calibrated)",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
_out = os.path.join(OUT_DIR, "reliability_diagrams.png")
fig.savefig(_out, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "reliability_diagrams.png"), bbox_inches="tight")
plt.close()
print(f"  Saved: {_out}")

# ===========================================================================
# 10. BRIER SCORE COMPARISON
# ===========================================================================
print("--- Brier Score Comparison ---")

brier_rows = results_df[results_df["Model"].isin(list(calib_store.keys()))].copy()
brier_rows = brier_rows.dropna(subset=["Brier", "Brier (Platt)", "Brier (Isotonic)"])
brier_rows = brier_rows.sort_values("Brier").reset_index(drop=True)

x  = np.arange(len(brier_rows))
w  = 0.25

fig, ax = plt.subplots(figsize=(max(14, len(brier_rows) * 1.3), 6))
ax.bar(x - w, brier_rows["Brier"],             w,
       label="Uncalibrated", color="#2563eb", alpha=0.85)
ax.bar(x,     brier_rows["Brier (Platt)"],     w,
       label="Platt Scaling", color="#ea580c", alpha=0.85)
ax.bar(x + w, brier_rows["Brier (Isotonic)"],  w,
       label="Isotonic Reg.", color="#16a34a", alpha=0.85)

ax.axhline(NAIVE_BRIER, color="red", linestyle="--", linewidth=1.5,
           alpha=0.7, label=f"Naive baseline ({NAIVE_BRIER:.4f})")

ax.set_xticks(x)
ax.set_xticklabels(brier_rows["Model"], rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Brier Score  (lower = better)", fontsize=10)
ax.set_title(
    "Brier Score Comparison: Uncalibrated vs Platt Scaling vs Isotonic Regression\n"
    f"Naive baseline (always predict prevalence {y_test.mean():.3f}) = {NAIVE_BRIER:.4f}",
    fontweight="bold", fontsize=12,
)
ax.legend(fontsize=10)
sns.despine()
plt.tight_layout()
_out = os.path.join(OUT_DIR, "brier_scores.png")
fig.savefig(_out, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "brier_scores.png"), bbox_inches="tight")
plt.close()
print(f"  Saved: {_out}")

# -- Calibration delta CSV -------------------------------------------------
delta_rows = []
for _, row in brier_rows.iterrows():
    delta_rows.append({
        "Model":            row["Model"],
        "Experiment":       row["Experiment"],
        "Brier (Raw)":      row["Brier"],
        "Brier (Platt)":    row["Brier (Platt)"],
        "Brier (Isotonic)": row["Brier (Isotonic)"],
        "ΔBrier Platt":     row["Brier (Platt)"]    - row["Brier"],
        "ΔBrier Isotonic":  row["Brier (Isotonic)"] - row["Brier"],
        "Best Calibration": ("Isotonic"
                             if row["Brier (Isotonic)"] < row["Brier (Platt)"]
                             else "Platt"),
    })
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(os.path.join(OUT_DIR, "brier_delta.csv"),
                index=False, float_format="%.4f")

print("\nBrier Delta Table:")
print(delta_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"
      if isinstance(x, float) else str(x)))

# ===========================================================================
# 11. MASTER SUMMARY FIGURE
#     4-panel: AUC-PR / AUC-ROC / F1 / Brier, coloured by experiment group
# ===========================================================================
print("\n--- Master Summary Figure ---")

EXP_COLORS = {
    "Exp-0 Baseline":   "#94a3b8",
    "Exp-1 ClassWeight":"#fb923c",
    "Exp-2 SMOTE":      "#2563eb",
    "Exp-3 ADASYN":     "#7c3aed",
    "Hyper Tuned":      "#16a34a",
    "Deep Learning":    "#dc2626",
}

_sorted2 = results_df.sort_values("AUC-PR", ascending=False).reset_index(drop=True)
colors2  = [EXP_COLORS.get(e, "#444") for e in _sorted2["Experiment"]]
x2 = np.arange(len(_sorted2))

fig, axes4 = plt.subplots(2, 2, figsize=(20, 13))

PANELS = [
    ("AUC-PR",  axes4[0, 0], False, "AUC-PR  (higher = better)"),
    ("AUC-ROC", axes4[0, 1], False, "AUC-ROC  (higher = better)"),
    ("F1 @ t*", axes4[1, 0], False, "F1 @ optimal threshold t*  (higher = better)"),
    ("Brier",   axes4[1, 1], True,  "Brier Score  (lower = better)"),
]

for metric, ax, invert, ylabel in PANELS:
    vals = _sorted2[metric].fillna(0).values
    bars = ax.bar(x2, vals, color=colors2, alpha=0.85, width=0.7)

    if invert:
        ax.axhline(NAIVE_BRIER, color="red", linestyle="--",
                   linewidth=1.5, alpha=0.65,
                   label=f"Naive ({NAIVE_BRIER:.4f})")
        ax.legend(fontsize=8)

    # Value labels on top
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=5.5, rotation=90)

    ax.set_xticks(x2)
    ax.set_xticklabels(_sorted2["Model"], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{metric}  (models sorted by AUC-PR)",
                 fontweight="bold", fontsize=10)
    if invert:
        # Flip so "best" is at top visually
        ax.invert_yaxis()
    sns.despine(ax=ax)

# Shared legend for experiment colours
legend_handles = [
    mpatches.Patch(facecolor=c, label=e, alpha=0.85)
    for e, c in EXP_COLORS.items()
]
fig.legend(
    handles=legend_handles, loc="lower center", ncol=len(EXP_COLORS),
    fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
)

fig.suptitle(
    "Master Experiment Summary\n"
    "All Models — Exp-0 (Baseline) → Exp-1 (ClassWeight) → Exp-2 (SMOTE) → "
    "Exp-3 (ADASYN) → Hyper-Tuned → Deep Learning",
    fontsize=13, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
_out = os.path.join(OUT_DIR, "master_summary_figure.png")
fig.savefig(_out, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "master_summary_figure.png"), bbox_inches="tight")
plt.close()
print(f"  Saved: {_out}")

# ===========================================================================
# 12. FINAL PRINT
# ===========================================================================
print("\n" + "=" * 72)
print("ALL OUTPUTS")
print("=" * 72)
print(f"  {OUT_DIR}/master_summary.csv")
print(f"  {OUT_DIR}/brier_delta.csv")
print(f"  {OUT_DIR}/reliability_diagrams.png")
print(f"  {OUT_DIR}/brier_scores.png")
print(f"  {OUT_DIR}/master_summary_figure.png")
print(f"  {FIG_DIR}/reliability_diagrams.png")
print(f"  {FIG_DIR}/brier_scores.png")
print(f"  {FIG_DIR}/master_summary_figure.png")
print("=" * 72)
