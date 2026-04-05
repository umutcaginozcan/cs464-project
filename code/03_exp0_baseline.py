"""
Exp-0: Baseline — No Imbalance Handling
CS 464 Stroke Prediction Project

Standard ERM: L = (1/N) Σ ℓ(yᵢ, f(xᵢ))
Every sample contributes equally to the loss. With 95% majority,
the gradient is dominated by class 0 → models predict "No Stroke" for all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import *

setup_plot_style()
EXP = "exp0_baseline"
OUT_DIR = os.path.join("outputs", "results", EXP)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP-0: BASELINE — NO IMBALANCE HANDLING")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = get_split(df)
preprocessor = get_preprocessor()

# No class weighting, no resampling
models = get_models(class_weight=False)
pipelines = build_pipelines(models, preprocessor, resampler=None)

print(f"\nTrain: {len(X_train)} (stroke={y_train.sum()})")
print(f"Test:  {len(X_test)} (stroke={y_test.sum()})\n")

cv_df, test_df, roc_data, pr_data, cm_data, prob_data = \
    run_experiment(pipelines, X_train, y_train, X_test, y_test)

save_results(cv_df, test_df, OUT_DIR)
plot_confusion_matrices(cm_data, os.path.join(OUT_DIR, "confusion_matrices.png"),
                        "Confusion Matrices — Exp-0: Baseline (No Handling)")
plot_roc_curves(roc_data, os.path.join(OUT_DIR, "roc_curves.png"),
                "ROC Curves — Exp-0: Baseline")
plot_pr_curves(pr_data, y_test, os.path.join(OUT_DIR, "pr_curves.png"),
               "PR Curves — Exp-0: Baseline")

print("\n" + "=" * 60)
print("EXP-0 COMPLETE")
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"All outputs → {OUT_DIR}/")
print("=" * 60)
