"""
Exp-1: Class Weighting — Cost-Sensitive Learning
CS 464 Stroke Prediction Project

Weighted ERM: L_w = (1/N) Σ w_yᵢ · ℓ(yᵢ, f(xᵢ))
  w₀ = N/(2·N₀) ≈ 0.53,  w₁ = N/(2·N₁) ≈ 10.26
Equivalent to reweighting empirical distribution Q(x,y) ∝ w_y · P_emp(x,y)
without adding synthetic data — preserves the data manifold.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import *

setup_plot_style()
EXP = "exp1_class_weight"
OUT_DIR = os.path.join("outputs", "results", EXP)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP-1: CLASS WEIGHTING — COST-SENSITIVE LEARNING")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = get_split(df)
preprocessor = get_preprocessor()

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw = n_neg / n_pos

print(f"\nClass weights: w₀={len(y_train)/(2*n_neg):.3f}, w₁={len(y_train)/(2*n_pos):.3f}")
print(f"XGBoost scale_pos_weight: {spw:.2f}\n")

models = get_models(class_weight=True, spw=spw)
pipelines = build_pipelines(models, preprocessor, resampler=None)

cv_df, test_df, roc_data, pr_data, cm_data, prob_data = \
    run_experiment(pipelines, X_train, y_train, X_test, y_test)

save_results(cv_df, test_df, OUT_DIR)
plot_confusion_matrices(cm_data, os.path.join(OUT_DIR, "confusion_matrices.png"),
                        "Confusion Matrices — Exp-1: Class Weighting")
plot_roc_curves(roc_data, os.path.join(OUT_DIR, "roc_curves.png"),
                "ROC Curves — Exp-1: Class Weighting")
plot_pr_curves(pr_data, y_test, os.path.join(OUT_DIR, "pr_curves.png"),
               "PR Curves — Exp-1: Class Weighting")

# ── Compare with Exp-0 ──
exp0_path = os.path.join("outputs", "results", "exp0_baseline", "test_results.csv")
if os.path.exists(exp0_path):
    exp0 = pd.read_csv(exp0_path)
    delta = test_df.set_index("Model")[["F1", "Recall", "AUC-PR"]].subtract(
        exp0.set_index("Model")[["F1", "Recall", "AUC-PR"]]
    ).reset_index()
    delta.columns = ["Model", "ΔF1", "ΔRecall", "ΔAUC-PR"]
    delta.to_csv(os.path.join(OUT_DIR, "delta_vs_exp0.csv"), index=False, float_format="%.4f")

    print("\n\nΔ vs Exp-0 (Class Weight − Baseline):")
    print(delta.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["F1", "Recall", "AUC-PR"]
    for ax, met in zip(axes, metrics):
        x = np.arange(len(MODEL_ORDER))
        w = 0.35
        v0 = exp0.set_index("Model").loc[MODEL_ORDER, met].values
        v1 = test_df.set_index("Model").loc[MODEL_ORDER, met].values
        ax.bar(x - w/2, v0, w, label="Exp-0: Baseline", color="#94a3b8")
        ax.bar(x + w/2, v1, w, label="Exp-1: Class Weight", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(met); ax.set_title(met)
        ax.legend(fontsize=8)
    sns.despine()
    fig.suptitle("Exp-0 vs Exp-1: Effect of Class Weighting", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "comparison_vs_exp0.png"))
    plt.close()

print("\n" + "=" * 60)
print("EXP-1 COMPLETE")
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"All outputs → {OUT_DIR}/")
print("=" * 60)
