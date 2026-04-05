"""
Shared utilities for Phase 2 experiments.
CS 464 Stroke Prediction Project

All experiments import from this module to ensure identical
data loading, splitting, preprocessing, and evaluation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════
SEED = 42
N_FOLDS = 5
DPI = 300
DATA_PATH = os.path.join("data", "healthcare-dataset-stroke-data.csv")

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
PASSTHROUGH = ["hypertension", "heart_disease", "bmi_missing"]

MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "Gaussian NB":         "#dc2626",
    "KNN (k=1)":           "#d4d4d4",
    "KNN (k=3)":           "#a3a3a3",
    "KNN (k=5)":           "#a3a3a3",
    "KNN (k=7)":           "#737373",
    "KNN (k=9)":           "#737373",
    "SVM (RBF)":           "#7c3aed",
    "Random Forest":       "#16a34a",
    "XGBoost":             "#ea580c",
}
MODEL_ORDER = list(MODEL_COLORS.keys())


def setup_plot_style():
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.facecolor": "white",
    })


# ══════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi_missing"] = df["bmi"].isna().astype(int)
    df.drop(columns=["id"], inplace=True)
    return df


def get_split(df):
    X = df[NUMERIC + CATEGORICAL + PASSTHROUGH]
    y = df["stroke"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def get_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("pass", "passthrough", PASSTHROUGH),
    ])


# ══════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════
def get_models(class_weight=False, spw=1.0):
    """Return dict of 10 models. class_weight enables built-in weighting."""
    cw = "balanced" if class_weight else None
    sw = spw if class_weight else 1.0
    return {
        "Logistic Regression": LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=SEED),
        "Gaussian NB": GaussianNB(
            priors=[0.5, 0.5] if class_weight else None),
        "KNN (k=1)": KNeighborsClassifier(n_neighbors=1),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "KNN (k=9)": KNeighborsClassifier(n_neighbors=9),
        "SVM (RBF)": SVC(
            class_weight=cw, probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight=cw, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, scale_pos_weight=sw,
            random_state=SEED, eval_metric="logloss", verbosity=0),
    }


def build_pipelines(models, preprocessor, resampler=None):
    """Wrap each model in Preprocessor [→ Resampler] → Model pipeline."""
    pipelines = {}
    for name, model in models.items():
        steps = [("preprocessor", clone(preprocessor))]
        if resampler is not None:
            steps.append(("resampler", clone(resampler)))
        steps.append(("model", model))
        pipelines[name] = ImbPipeline(steps)
    return pipelines


# ══════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════
def run_experiment(pipelines, X_train, y_train, X_test, y_test, verbose=True):
    """Run 5-fold CV + held-out test evaluation.
    Returns (cv_df, test_df, roc_data, pr_data, cm_data, prob_data).
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ── Cross-Validation ──
    cv_records = []
    for name, pipe in pipelines.items():
        if verbose:
            print(f"  CV: {name} ...", end=" ", flush=True)
        fold_metrics = {k: [] for k in
                        ["accuracy", "f1", "recall", "precision", "roc_auc", "auc_pr"]}
        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            try:
                p = clone(pipe)
                p.fit(X_tr, y_tr)
                y_pred = p.predict(X_val)
                y_prob = p.predict_proba(X_val)[:, 1]
                fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
                fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
                fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
                fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
                fold_metrics["auc_pr"].append(average_precision_score(y_val, y_prob))
            except Exception as e:
                if verbose:
                    print(f"\n    ⚠ Fold {fold_idx} failed: {e}")
                for k in fold_metrics:
                    fold_metrics[k].append(np.nan)

        rec = {"Model": name}
        for mk, dk in [("Accuracy", "accuracy"), ("F1", "f1"), ("Recall", "recall"),
                        ("Precision", "precision"), ("AUC-ROC", "roc_auc"), ("AUC-PR", "auc_pr")]:
            vals = np.array(fold_metrics[dk])
            rec[f"{mk} (mean)"] = np.nanmean(vals)
            rec[f"{mk} (std)"] = np.nanstd(vals)
        cv_records.append(rec)
        if verbose:
            print(f"AUC-PR={rec['AUC-PR (mean)']:.4f} ± {rec['AUC-PR (std)']:.4f}")
    cv_df = pd.DataFrame(cv_records)

    # ── Held-out Test ──
    test_records, roc_data, pr_data, cm_data, prob_data = [], {}, {}, {}, {}
    for name, pipe in pipelines.items():
        if verbose:
            print(f"  Test: {name} ...", end=" ", flush=True)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        ap  = average_precision_score(y_test, y_prob)

        test_records.append({"Model": name, "Accuracy": acc, "F1": f1, "Recall": rec,
                             "Precision": prec, "AUC-ROC": roc, "AUC-PR": ap})
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc)
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (rec_c, prec_c, ap)
        cm_data[name] = confusion_matrix(y_test, y_pred)
        prob_data[name] = y_prob
        if verbose:
            print(f"AUC-PR={ap:.4f}, Recall={rec:.4f}")

    test_df = pd.DataFrame(test_records)
    return cv_df, test_df, roc_data, pr_data, cm_data, prob_data


# ══════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════
def plot_confusion_matrices(cm_data, out_path, suptitle="Confusion Matrices"):
    n = len(cm_data)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    for i, (name, cm) in enumerate(cm_data.items()):
        ax = axes[i]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm_norm[r,c]:.1f}%\n(n={cm[r,c]})" for c in range(2)] for r in range(2)])
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        for r in range(2):
            for c in range(2):
                color = "white" if cm_norm[r, c] > 60 else "black"
                ax.text(c, r, annot[r, c], ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Stroke", "Stroke"]); ax.set_yticklabels(["No Stroke", "Stroke"])
        ax.set_title(name, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.005)
    fig.tight_layout(); fig.savefig(out_path); plt.close()


def plot_roc_curves(roc_data, out_path, title="ROC Curves"):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (fpr, tpr, auc_val) in sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True):
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, "#333"), linewidth=2.5,
                label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title,
           xlim=[0, 1], ylim=[0, 1.02], aspect="equal")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    sns.despine(); fig.tight_layout(); fig.savefig(out_path); plt.close()


def plot_pr_curves(pr_data, y_test, out_path, title="PR Curves"):
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()
    for name, (rec_c, prec_c, ap) in sorted(pr_data.items(), key=lambda x: x[1][2], reverse=True):
        ax.plot(rec_c, prec_c, color=MODEL_COLORS.get(name, "#333"), linewidth=2.5,
                label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=baseline, color="k", ls="--", alpha=0.3, lw=1,
               label=f"Prevalence ({baseline:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title,
           xlim=[0, 1], ylim=[0, 1.02], aspect="equal")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    sns.despine(); fig.tight_layout(); fig.savefig(out_path); plt.close()


def save_results(cv_df, test_df, out_dir):
    cv_df.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False, float_format="%.4f")
    test_df.to_csv(os.path.join(out_dir, "test_results.csv"), index=False, float_format="%.4f")
