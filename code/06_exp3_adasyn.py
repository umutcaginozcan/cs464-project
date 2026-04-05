"""
Exp-3: ADASYN — Adaptive Synthetic Oversampling
CS 464 Stroke Prediction Project

ADASYN replaces SMOTE's uniform seed selection with adaptive weighting:
  Γᵢ = |{x_j ∈ kNN(xᵢ) : y_j = 0}| / k     (majority-neighbor ratio)
  r̂ᵢ = Γᵢ / Σ Γⱼ                              (normalized density)
  P(select xᵢ) = r̂ᵢ   instead of 1/N_min

Hypothesis: if Γᵢ ≈ Γⱼ ∀ i,j → r̂ᵢ ≈ 1/N_min → ADASYN ≈ SMOTE.
We test this by computing the Γᵢ distribution on our dataset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import *
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import NearestNeighbors

setup_plot_style()
EXP = "exp3_adasyn"
OUT_DIR = os.path.join("outputs", "results", EXP)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP-3: ADASYN — ADAPTIVE SYNTHETIC OVERSAMPLING")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = get_split(df)
preprocessor = get_preprocessor()

# No class weight — isolate ADASYN effect
models = get_models(class_weight=False)
pipelines = build_pipelines(models, preprocessor, resampler=ADASYN(random_state=SEED))

cv_df, test_df, roc_data, pr_data, cm_data, prob_data = \
    run_experiment(pipelines, X_train, y_train, X_test, y_test)

save_results(cv_df, test_df, OUT_DIR)
plot_confusion_matrices(cm_data, os.path.join(OUT_DIR, "confusion_matrices.png"),
                        "Confusion Matrices — Exp-3: ADASYN")
plot_roc_curves(roc_data, os.path.join(OUT_DIR, "roc_curves.png"),
                "ROC Curves — Exp-3: ADASYN")
plot_pr_curves(pr_data, y_test, os.path.join(OUT_DIR, "pr_curves.png"),
               "PR Curves — Exp-3: ADASYN")

# ── Compare ADASYN vs SMOTE (Exp-2) ──
exp2_path = os.path.join("outputs", "results", "exp2_smote", "test_results.csv")
if os.path.exists(exp2_path):
    exp2 = pd.read_csv(exp2_path)
    delta = test_df.set_index("Model")[["F1", "Recall", "AUC-PR"]].subtract(
        exp2.set_index("Model")[["F1", "Recall", "AUC-PR"]]
    ).reset_index()
    delta.columns = ["Model", "ΔF1", "ΔRecall", "ΔAUC-PR"]
    delta.to_csv(os.path.join(OUT_DIR, "delta_vs_smote.csv"), index=False, float_format="%.4f")
    print("\n\nΔ vs Exp-2 (ADASYN − SMOTE):")
    print(delta.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # Side-by-side bar chart: SMOTE vs ADASYN
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, met in zip(axes, ["F1", "Recall", "AUC-PR"]):
        x = np.arange(len(MODEL_ORDER))
        w = 0.35
        v_smote  = exp2.set_index("Model").loc[MODEL_ORDER, met].values
        v_adasyn = test_df.set_index("Model").loc[MODEL_ORDER, met].values
        ax.bar(x - w/2, v_smote, w, label="SMOTE (uniform)", color="#ea580c")
        ax.bar(x + w/2, v_adasyn, w, label="ADASYN (adaptive)", color="#7c3aed")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(met); ax.set_title(met)
        ax.legend(fontsize=8)
    sns.despine()
    fig.suptitle("SMOTE vs ADASYN: Uniform vs Adaptive Oversampling",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "smote_vs_adasyn.png"))
    plt.close()

# ══════════════════════════════════════════════
# Γᵢ Distribution Analysis
# ══════════════════════════════════════════════
print("\n\n── Γᵢ Distribution Analysis ──")
print("Computing majority-neighbor ratio for each minority sample...\n")

# Transform training data
prep = get_preprocessor()
X_train_t = prep.fit_transform(X_train)
y_arr = y_train.values

minority_mask = y_arr == 1
X_min = X_train_t[minority_mask]
N_min = minority_mask.sum()

# kNN on full training set (ADASYN default k=5)
k = 5
nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to exclude self
nn.fit(X_train_t)
_, indices = nn.kneighbors(X_min)

gammas = []
for i in range(N_min):
    neighbors = indices[i, 1:]  # exclude self
    n_majority = (y_arr[neighbors] == 0).sum()
    gammas.append(n_majority / k)

gammas = np.array(gammas)
r_hat = gammas / gammas.sum()  # normalized ADASYN weights
uniform = 1.0 / N_min

print(f"  N_minority = {N_min}")
print(f"  Γᵢ  mean = {gammas.mean():.3f}, std = {gammas.std():.3f}")
print(f"  Γᵢ  median = {np.median(gammas):.3f}")
print(f"  Γᵢ ≥ 0.8: {(gammas >= 0.8).sum()} / {N_min} ({(gammas >= 0.8).mean()*100:.1f}%)")
print(f"  r̂ᵢ  std  = {r_hat.std():.6f}  (uniform = {uniform:.6f})")
print(f"  r̂ᵢ  CV   = {r_hat.std()/r_hat.mean():.3f}  → {'low variance → ADASYN ≈ SMOTE' if r_hat.std()/r_hat.mean() < 0.5 else 'high variance → ADASYN ≠ SMOTE'}")

# Save Γ stats
gamma_stats = pd.DataFrame({
    "Metric": ["N_minority", "Γ_mean", "Γ_std", "Γ_median",
               "Γ≥0.8 (count)", "Γ≥0.8 (%)", "r_hat_std", "r_hat_CV", "1/N_min"],
    "Value": [N_min, gammas.mean(), gammas.std(), np.median(gammas),
              (gammas >= 0.8).sum(), (gammas >= 0.8).mean()*100,
              r_hat.std(), r_hat.std()/r_hat.mean(), uniform]
})
gamma_stats.to_csv(os.path.join(OUT_DIR, "gamma_distribution_stats.csv"),
                   index=False, float_format="%.6f")

# Plot Γᵢ histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Γᵢ histogram
axes[0].hist(gammas, bins=np.arange(0, 1.1, 0.1), color="#7c3aed",
             edgecolor="white", alpha=0.8)
axes[0].axvline(x=gammas.mean(), color="red", ls="--", lw=2,
                label=f"Mean Γ = {gammas.mean():.2f}")
axes[0].set_xlabel("Γᵢ (majority-neighbor ratio)")
axes[0].set_ylabel("Count of minority samples")
axes[0].set_title("Distribution of Γᵢ (ADASYN difficulty score)")
axes[0].legend()

# Right: r̂ᵢ vs uniform
sorted_r = np.sort(r_hat)[::-1]
axes[1].bar(range(N_min), sorted_r, color="#7c3aed", alpha=0.7, width=1.0)
axes[1].axhline(y=uniform, color="red", ls="--", lw=2,
                label=f"Uniform = 1/N = {uniform:.5f}")
axes[1].set_xlabel("Minority sample (sorted by r̂ᵢ)")
axes[1].set_ylabel("Selection probability r̂ᵢ")
axes[1].set_title("ADASYN weights vs SMOTE uniform weights")
axes[1].legend()

sns.despine()
fig.suptitle("Why ADASYN ≈ SMOTE on this dataset: Γᵢ is nearly constant",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gamma_distribution.png"))
plt.close()

print("\n" + "=" * 60)
print("EXP-3 COMPLETE")
print(test_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"All outputs → {OUT_DIR}/")
print("=" * 60)
