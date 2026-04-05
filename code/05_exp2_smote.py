"""
Exp-2: SMOTE — Synthetic Minority Oversampling
CS 464 Stroke Prediction Project

SMOTE generates synthetic minority samples via linear interpolation:
  x_new = xᵢ + λ·(x_j − xᵢ),  λ ~ U(0,1),  x_j ∈ kNN(xᵢ)

Key assumption: P(select xᵢ as seed) = 1/N_min  (UNIFORM)
Every minority sample is equally likely to be chosen — no distinction
between "easy" interior points and "hard" boundary points.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import *
from imblearn.over_sampling import SMOTE

setup_plot_style()
EXP = "exp2_smote"
OUT_DIR = os.path.join("outputs", "results", EXP)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP-2: SMOTE — UNIFORM SYNTHETIC OVERSAMPLING")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = get_split(df)
preprocessor = get_preprocessor()

# No class weight — isolate SMOTE effect
models = get_models(class_weight=False)
pipelines = build_pipelines(models, preprocessor, resampler=SMOTE(random_state=SEED))

cv_df, test_df, roc_data, pr_data, cm_data, prob_data = \
    run_experiment(pipelines, X_train, y_train, X_test, y_test)

save_results(cv_df, test_df, OUT_DIR)
plot_confusion_matrices(cm_data, os.path.join(OUT_DIR, "confusion_matrices.png"),
                        "Confusion Matrices — Exp-2: SMOTE")
plot_roc_curves(roc_data, os.path.join(OUT_DIR, "roc_curves.png"),
                "ROC Curves — Exp-2: SMOTE")
plot_pr_curves(pr_data, y_test, os.path.join(OUT_DIR, "pr_curves.png"),
               "PR Curves — Exp-2: SMOTE")

# ── Compare with Exp-0 ──
exp0_path = os.path.join("outputs", "results", "exp0_baseline", "test_results.csv")
if os.path.exists(exp0_path):
    exp0 = pd.read_csv(exp0_path)
    delta = test_df.set_index("Model")[["F1", "Recall", "AUC-PR"]].subtract(
        exp0.set_index("Model")[["F1", "Recall", "AUC-PR"]]
    ).reset_index()
    delta.columns = ["Model", "ΔF1", "ΔRecall", "ΔAUC-PR"]
    delta.to_csv(os.path.join(OUT_DIR, "delta_vs_exp0.csv"), index=False, float_format="%.4f")
    print("\n\nΔ vs Exp-0 (SMOTE − Baseline):")
    print(delta.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Grouped bar: Exp-0 vs Exp-1 vs Exp-2
    exp1_path = os.path.join("outputs", "results", "exp1_class_weight", "test_results.csv")
    if os.path.exists(exp1_path):
        exp1 = pd.read_csv(exp1_path)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ["F1", "Recall", "AUC-PR"]
        for ax, met in zip(axes, metrics):
            x = np.arange(len(MODEL_ORDER))
            w = 0.25
            v0 = exp0.set_index("Model").loc[MODEL_ORDER, met].values
            v1 = exp1.set_index("Model").loc[MODEL_ORDER, met].values
            v2 = test_df.set_index("Model").loc[MODEL_ORDER, met].values
            ax.bar(x - w, v0, w, label="Exp-0: Baseline", color="#94a3b8")
            ax.bar(x,     v1, w, label="Exp-1: Class Weight", color="#2563eb")
            ax.bar(x + w, v2, w, label="Exp-2: SMOTE", color="#ea580c")
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(met); ax.set_title(met)
            ax.legend(fontsize=7)
        sns.despine()
        fig.suptitle("Exp-0 vs Exp-1 vs Exp-2: Baseline → Class Weight → SMOTE",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "comparison_3way.png"))
        plt.close()

print("\n" + "=" * 60)
print("EXP-2 COMPLETE")
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"All outputs → {OUT_DIR}/")
print("=" * 60)
