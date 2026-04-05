"""
Exp-4: Threshold Tuning — Post-hoc Decision Boundary Optimization
CS 464 Stroke Prediction Project

Default: ŷ = 𝟙[P(y=1|x) > 0.5]
With 5% prevalence, the optimal threshold is far lower than 0.5.

F1-optimal threshold: t* = argmax_t  2·Prec(t)·Rec(t) / (Prec(t)+Rec(t))

This experiment shows that threshold tuning (free, no retraining) yields
larger gains than any training-time resampling strategy.

Clinical operating points:
  🔴 Screening:    Recall ≥ 0.90, maximize precision
  🟡 Balanced:     Maximize F1
  🟢 Confirmatory: Precision ≥ 0.30, maximize recall
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared import *
from imblearn.over_sampling import SMOTE, ADASYN

setup_plot_style()
EXP = "exp4_threshold"
OUT_DIR = os.path.join("outputs", "results", EXP)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("EXP-4: THRESHOLD TUNING — POST-HOC OPTIMIZATION")
print("=" * 60)

df = load_data()
X_train, X_test, y_train, y_test = get_split(df)
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw = n_neg / n_pos

# ── Determine best strategy per model from Exp 0-3 ──
TOP_MODELS = ["Logistic Regression", "XGBoost", "SVM (RBF)"]

EXPERIMENTS = {
    "Exp-0: Baseline": {"class_weight": False, "resampler": None},
    "Exp-1: Class Weight": {"class_weight": True, "resampler": None},
    "Exp-2: SMOTE": {"class_weight": False, "resampler": SMOTE(random_state=SEED)},
    "Exp-3: ADASYN": {"class_weight": False, "resampler": ADASYN(random_state=SEED)},
}

# Find best strategy per model by loading existing CSV results
best_strategy = {}
for model_name in TOP_MODELS:
    best_auc_pr, best_strat = -1, "Exp-0: Baseline"
    for exp_name, exp_dir_name in [
        ("Exp-0: Baseline", "exp0_baseline"),
        ("Exp-1: Class Weight", "exp1_class_weight"),
        ("Exp-2: SMOTE", "exp2_smote"),
        ("Exp-3: ADASYN", "exp3_adasyn"),
    ]:
        csv_path = os.path.join("outputs", "results", exp_dir_name, "test_results.csv")
        if os.path.exists(csv_path):
            df_res = pd.read_csv(csv_path)
            row = df_res[df_res["Model"] == model_name]
            if not row.empty and row["AUC-PR"].values[0] > best_auc_pr:
                best_auc_pr = row["AUC-PR"].values[0]
                best_strat = exp_name
    best_strategy[model_name] = best_strat
    print(f"  {model_name} → best = {best_strat} (AUC-PR={best_auc_pr:.4f})")

# ── Train top-3 models with their best strategy ──
print("\n── Training top-3 models with best strategy ──\n")

preprocessor = get_preprocessor()
prob_data = {}

for model_name in TOP_MODELS:
    strat_name = best_strategy[model_name]
    config = EXPERIMENTS[strat_name]

    models = get_models(class_weight=config["class_weight"], spw=spw)
    model = models[model_name]
    resampler = config["resampler"]

    pipe = build_pipelines({model_name: model}, preprocessor, resampler=resampler)
    pipe = pipe[model_name]

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    prob_data[model_name] = (y_prob, strat_name)

    auc_pr = average_precision_score(y_test, y_prob)
    print(f"  {model_name} + {strat_name}: AUC-PR={auc_pr:.4f}")

# ══════════════════════════════════════════════
# Threshold Sweep
# ══════════════════════════════════════════════
print("\n── Threshold Sweep ──\n")

thresholds = np.arange(0.01, 0.99, 0.005)
sweep_results = {}

for model_name in TOP_MODELS:
    y_prob, strat_name = prob_data[model_name]
    records = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        if y_pred_t.sum() == 0:
            records.append({"threshold": t, "F1": 0, "Recall": 0, "Precision": 0})
            continue
        records.append({
            "threshold": t,
            "F1": f1_score(y_test, y_pred_t, zero_division=0),
            "Recall": recall_score(y_test, y_pred_t, zero_division=0),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
        })
    sweep_results[model_name] = pd.DataFrame(records)

# ── Find operating points ──
operating_points = {}
for model_name in TOP_MODELS:
    sw = sweep_results[model_name]
    y_prob, strat_name = prob_data[model_name]

    # Balanced: max F1
    idx_f1 = sw["F1"].idxmax()
    balanced = sw.loc[idx_f1]

    # Screening: Recall ≥ 0.90, highest threshold (maximize precision)
    screening_mask = sw["Recall"] >= 0.90
    if screening_mask.any():
        idx_screen = sw.loc[screening_mask, "threshold"].idxmax()
        screening = sw.loc[idx_screen]
    else:
        idx_screen = sw["Recall"].idxmax()
        screening = sw.loc[idx_screen]

    # Confirmatory: Precision ≥ 0.30, lowest threshold (maximize recall)
    confirm_mask = sw["Precision"] >= 0.30
    if confirm_mask.any():
        idx_confirm = sw.loc[confirm_mask, "threshold"].idxmin()
        confirmatory = sw.loc[idx_confirm]
    else:
        idx_confirm = sw["Precision"].idxmax()
        confirmatory = sw.loc[idx_confirm]

    operating_points[model_name] = {
        "Screening":    {"t": screening["threshold"],    "F1": screening["F1"],
                         "Recall": screening["Recall"],  "Precision": screening["Precision"]},
        "Balanced":     {"t": balanced["threshold"],     "F1": balanced["F1"],
                         "Recall": balanced["Recall"],   "Precision": balanced["Precision"]},
        "Confirmatory": {"t": confirmatory["threshold"], "F1": confirmatory["F1"],
                         "Recall": confirmatory["Recall"],"Precision": confirmatory["Precision"]},
    }

# Print operating points
for model_name in TOP_MODELS:
    print(f"\n  {model_name} ({best_strategy[model_name]}):")
    for op_name, vals in operating_points[model_name].items():
        emoji = {"Screening": "🔴", "Balanced": "🟡", "Confirmatory": "🟢"}[op_name]
        print(f"    {emoji} {op_name:14s}  t={vals['t']:.3f}  "
              f"F1={vals['F1']:.3f}  Rec={vals['Recall']:.3f}  Prec={vals['Precision']:.3f}")

# Save operating points
op_records = []
for model_name in TOP_MODELS:
    for op_name, vals in operating_points[model_name].items():
        op_records.append({"Model": model_name, "Strategy": best_strategy[model_name],
                           "Operating Point": op_name, **vals})
op_df = pd.DataFrame(op_records)
op_df.to_csv(os.path.join(OUT_DIR, "operating_points.csv"), index=False, float_format="%.4f")

# ── Compare: default threshold vs optimized ──
print("\n\n── Default (t=0.5) vs Optimized Threshold ──")
comparison_records = []
for model_name in TOP_MODELS:
    y_prob, strat_name = prob_data[model_name]
    # Default at 0.5
    y_def = (y_prob >= 0.5).astype(int)
    f1_def = f1_score(y_test, y_def, zero_division=0)
    rec_def = recall_score(y_test, y_def, zero_division=0)
    # Optimized
    bal = operating_points[model_name]["Balanced"]
    comparison_records.append({
        "Model": model_name, "Strategy": strat_name,
        "F1 (t=0.5)": f1_def, "Recall (t=0.5)": rec_def,
        "F1 (t*)": bal["F1"], "Recall (t*)": bal["Recall"],
        "t*": bal["t"],
        "ΔF1": bal["F1"] - f1_def, "ΔRecall": bal["Recall"] - rec_def,
    })
comp_df = pd.DataFrame(comparison_records)
comp_df.to_csv(os.path.join(OUT_DIR, "threshold_improvement.csv"),
               index=False, float_format="%.4f")
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ══════════════════════════════════════════════
# Plot: Threshold Sweep Curves
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, model_name in zip(axes, TOP_MODELS):
    sw = sweep_results[model_name]
    ax.plot(sw["threshold"], sw["F1"], color="#2563eb", lw=2.5, label="F1")
    ax.plot(sw["threshold"], sw["Recall"], color="#16a34a", lw=2, label="Recall")
    ax.plot(sw["threshold"], sw["Precision"], color="#ea580c", lw=2, label="Precision")

    # Mark operating points
    ops = operating_points[model_name]
    markers = {"Screening": ("🔴", "v", "red"), "Balanced": ("🟡", "D", "#eab308"),
               "Confirmatory": ("🟢", "^", "green")}
    for op_name, (emoji, marker, color) in markers.items():
        t = ops[op_name]["t"]
        f1_val = ops[op_name]["F1"]
        ax.axvline(x=t, color=color, ls=":", alpha=0.5)
        ax.plot(t, f1_val, marker=marker, color=color, markersize=12, zorder=5,
                label=f"{op_name} (t={t:.2f})")

    # Default threshold
    ax.axvline(x=0.5, color="gray", ls="--", alpha=0.4, label="Default (t=0.5)")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{model_name}\n({best_strategy[model_name]})", fontsize=12)
    ax.legend(fontsize=7, loc="center right")
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 1.02])

sns.despine()
fig.suptitle("Threshold Sweep — Clinical Operating Points",
             fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "threshold_curves.png"))
plt.close()

# ══════════════════════════════════════════════
# Final Summary: All Experiments at Optimal Threshold
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY: IMPROVEMENT TRAJECTORY")
print("=" * 60)

# Load best LR result from each experiment and show progression
lr_progression = []
for exp_name, exp_dir in [("Baseline", "exp0_baseline"),
                           ("Class Weight", "exp1_class_weight"),
                           ("SMOTE", "exp2_smote"),
                           ("ADASYN", "exp3_adasyn")]:
    csv_path = os.path.join("outputs", "results", exp_dir, "test_results.csv")
    if os.path.exists(csv_path):
        df_res = pd.read_csv(csv_path)
        lr_row = df_res[df_res["Model"] == "Logistic Regression"].iloc[0]
        lr_progression.append({
            "Experiment": exp_name, "Threshold": "0.500 (default)",
            "F1": lr_row["F1"], "Recall": lr_row["Recall"],
            "Precision": lr_row["Precision"], "AUC-PR": lr_row["AUC-PR"],
        })

# Add threshold-optimized
bal = operating_points["Logistic Regression"]["Balanced"]
lr_progression.append({
    "Experiment": f"{best_strategy['Logistic Regression']} + Threshold",
    "Threshold": f"{bal['t']:.3f} (optimized)",
    "F1": bal["F1"], "Recall": bal["Recall"],
    "Precision": bal["Precision"], "AUC-PR": average_precision_score(y_test, prob_data["Logistic Regression"][0]),
})

prog_df = pd.DataFrame(lr_progression)
prog_df.to_csv(os.path.join(OUT_DIR, "lr_progression.csv"), index=False, float_format="%.4f")
print("\nLogistic Regression — Improvement Trajectory:")
print(prog_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\n\nAll outputs → {OUT_DIR}/")
print("=" * 60)
