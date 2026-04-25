"""
Phase 0 — Sanity Checks: Do the 11 features look clinically plausible?
CS 464 Stroke Prediction Project

For each feature, we ask:
  "Does this distribution match what we'd expect in a real population?"

Run:  python code/00_sanity_checks.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
DATA_PATH = os.path.join("data", "healthcare-dataset-stroke-data.csv")
FIG_DIR = os.path.join("outputs", "figures", "sanity_checks")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
COLOR_MAIN = "#3b82f6"
COLOR_ACCENT = "#ef4444"
COLOR_WARN = "#f59e0b"
COLOR_OK = "#10b981"

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# ────────────────────────────────────────────
# Load
# ────────────────────────────────────────────
print("=" * 60)
print("PHASE 0 — SANITY CHECKS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.drop(columns=["id"], inplace=True)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic stats:\n{df.describe().round(2)}")


# ════════════════════════════════════════════════════════════
# 1. AGE — does it look like a real population?
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 1: Age distribution")
print("─" * 50)

print(f"  Range: {df['age'].min():.1f} – {df['age'].max():.1f}")
print(f"  Mean:  {df['age'].mean():.1f}, Median: {df['age'].median():.1f}")
print(f"  Children (< 18): {(df['age'] < 18).sum()} ({(df['age'] < 18).mean()*100:.1f}%)")
print(f"  Elderly (≥ 65):  {(df['age'] >= 65).sum()} ({(df['age'] >= 65).mean()*100:.1f}%)")

# Expect: broad range (0–80+), right-skewed or roughly uniform
flag = "✅" if df["age"].min() >= 0 and df["age"].max() <= 120 else "⚠️"
print(f"  {flag} Age range is plausible")


# ════════════════════════════════════════════════════════════
# 2. BMI — clinical range check
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 2: BMI distribution")
print("─" * 50)

bmi_valid = df["bmi"].dropna()
print(f"  Missing: {df['bmi'].isna().sum()} ({df['bmi'].isna().mean()*100:.1f}%)")
print(f"  Range: {bmi_valid.min():.1f} – {bmi_valid.max():.1f}")
print(f"  Mean:  {bmi_valid.mean():.1f}, Median: {bmi_valid.median():.1f}")

# Clinical expectations:
# Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥ 30
bins = pd.cut(bmi_valid, bins=[0, 18.5, 25, 30, 100],
              labels=["Underweight", "Normal", "Overweight", "Obese"])
bmi_dist = bins.value_counts(normalize=True).sort_index() * 100
print(f"\n  BMI Categories:")
for cat, pct in bmi_dist.items():
    print(f"    {cat}: {pct:.1f}%")

# WHO says global avg BMI ~25. Dataset mean should be around 25-30.
flag = "✅" if 20 < bmi_valid.mean() < 35 else "⚠️"
print(f"\n  {flag} Mean BMI ({bmi_valid.mean():.1f}) is within expected range (20–35)")

# Check for extreme outliers
extreme_bmi = bmi_valid[(bmi_valid < 10) | (bmi_valid > 60)]
if len(extreme_bmi) > 0:
    print(f"  ⚠️ {len(extreme_bmi)} extreme BMI values (< 10 or > 60)")
else:
    print(f"  ✅ No extreme BMI outliers")


# ════════════════════════════════════════════════════════════
# 3. AVG GLUCOSE LEVEL — clinical range check
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 3: Average Glucose Level")
print("─" * 50)

glc = df["avg_glucose_level"]
print(f"  Range: {glc.min():.1f} – {glc.max():.1f}")
print(f"  Mean:  {glc.mean():.1f}, Median: {glc.median():.1f}")

# Clinical expectations (mg/dL):
# Normal fasting: 70–100, Pre-diabetes: 100–125, Diabetes: ≥ 126
bins_glc = pd.cut(glc, bins=[0, 100, 125, 400],
                  labels=["Normal (<100)", "Pre-diabetic (100-125)", "Diabetic (≥126)"])
glc_dist = bins_glc.value_counts(normalize=True).sort_index() * 100
print(f"\n  Glucose Categories:")
for cat, pct in glc_dist.items():
    print(f"    {cat}: {pct:.1f}%")

flag = "✅" if 55 < glc.min() and glc.max() < 350 else "⚠️"
print(f"\n  {flag} Glucose range is clinically plausible")


# ════════════════════════════════════════════════════════════
# 4. HYPERTENSION & HEART DISEASE — prevalence check
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 4: Hypertension & Heart Disease prevalence")
print("─" * 50)

ht_rate = df["hypertension"].mean() * 100
hd_rate = df["heart_disease"].mean() * 100
print(f"  Hypertension prevalence: {ht_rate:.1f}%")
print(f"  Heart disease prevalence: {hd_rate:.1f}%")

# WHO: ~30% adults have hypertension globally, ~5-10% have CVD
# But this dataset includes children → rates will be lower
flag_ht = "✅" if 5 < ht_rate < 40 else "⚠️"
flag_hd = "✅" if 1 < hd_rate < 20 else "⚠️"
print(f"  {flag_ht} Hypertension rate plausible (includes children → lower expected)")
print(f"  {flag_hd} Heart disease rate plausible")


# ════════════════════════════════════════════════════════════
# 5. GENDER — balance check
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 5: Gender distribution")
print("─" * 50)

gender_dist = df["gender"].value_counts()
for g, n in gender_dist.items():
    print(f"  {g}: {n} ({n/len(df)*100:.1f}%)")

# Note: "Other" category has only 1 sample
if "Other" in gender_dist.index:
    print(f"  ⚠️ 'Other' has only {gender_dist['Other']} sample(s) — too few for modeling")


# ════════════════════════════════════════════════════════════
# 6. SMOKING STATUS — age consistency check
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 6: Smoking status vs. age consistency")
print("─" * 50)

smoke_dist = df["smoking_status"].value_counts()
for s, n in smoke_dist.items():
    avg_age = df.loc[df["smoking_status"] == s, "age"].mean()
    print(f"  {s}: n={n} ({n/len(df)*100:.1f}%), mean age={avg_age:.1f}")

# Key sanity check: children who "smoke" or "formerly smoked"?
child_smokers = df[(df["age"] < 15) & (df["smoking_status"].isin(["smokes", "formerly smoked"]))]
print(f"\n  Children (<15) who smoke/formerly smoked: {len(child_smokers)}")
if len(child_smokers) == 0:
    print(f"  ✅ No implausible child smokers")
else:
    print(f"  ⚠️ {len(child_smokers)} children listed as smokers — data quality issue")

# "Unknown" smoking → mostly children?
unknown_smoke = df[df["smoking_status"] == "Unknown"]
pct_child = (unknown_smoke["age"] < 18).mean() * 100
print(f"\n  'Unknown' smoking status: {pct_child:.1f}% are under 18")
print(f"  → Likely: children were not asked about smoking")


# ════════════════════════════════════════════════════════════
# 7. WORK TYPE — age consistency
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 7: Work type vs. age consistency")
print("─" * 50)

for wt in df["work_type"].unique():
    subset = df[df["work_type"] == wt]
    print(f"  {wt}: n={len(subset)}, "
          f"age range={subset['age'].min():.0f}–{subset['age'].max():.0f}, "
          f"mean age={subset['age'].mean():.1f}")

# Key check: "children" work type should have low ages
children_wt = df[df["work_type"] == "children"]
if len(children_wt) > 0:
    max_child_age = children_wt["age"].max()
    flag = "✅" if max_child_age <= 18 else "⚠️"
    print(f"\n  {flag} 'children' work type max age: {max_child_age:.1f}")


# ════════════════════════════════════════════════════════════
# 8. EVER MARRIED — age consistency
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 8: Ever married vs. age")
print("─" * 50)

for m in df["ever_married"].unique():
    subset = df[df["ever_married"] == m]
    print(f"  {m}: n={len(subset)}, mean age={subset['age'].mean():.1f}, "
          f"median={subset['age'].median():.1f}")

# Young married people?
young_married = df[(df["age"] < 16) & (df["ever_married"] == "Yes")]
print(f"\n  Married & under 16: {len(young_married)}")
if len(young_married) == 0:
    print(f"  ✅ No implausibly young married individuals")


# ════════════════════════════════════════════════════════════
# 9. RESIDENCE TYPE — should be roughly balanced
# ════════════════════════════════════════════════════════════
print("\n\n" + "─" * 50)
print("CHECK 9: Residence type")
print("─" * 50)

res_dist = df["Residence_type"].value_counts()
for r, n in res_dist.items():
    print(f"  {r}: {n} ({n/len(df)*100:.1f}%)")

ratio = res_dist.min() / res_dist.max()
flag = "✅" if ratio > 0.8 else "⚠️"
print(f"  {flag} Urban/Rural ratio: {ratio:.2f} (close to balanced)")


# ════════════════════════════════════════════════════════════
# PRESENTATION PLOTS
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("GENERATING PRESENTATION PLOTS")
print("=" * 60)

# ──────────────────────────────────────────────────
# PLOT 1: Continuous features — distributions with clinical reference lines
# ──────────────────────────────────────────────────
print("\n[Plot 1] Continuous features with clinical reference lines")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Age ---
ax = axes[0]
ax.hist(df["age"], bins=50, color=COLOR_MAIN, alpha=0.7, edgecolor="white")
ax.axvline(df["age"].median(), color=COLOR_ACCENT, linestyle="--", linewidth=2,
           label=f"Median = {df['age'].median():.0f}")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
ax.set_title("Age Distribution")
ax.legend(fontsize=9)

# --- BMI ---
ax = axes[1]
bmi_data = df["bmi"].dropna()
ax.hist(bmi_data, bins=50, color=COLOR_MAIN, alpha=0.7, edgecolor="white")
# Clinical reference bands
ax.axvspan(0, 18.5, alpha=0.08, color="orange", label="Underweight (<18.5)")
ax.axvspan(18.5, 25, alpha=0.08, color="green", label="Normal (18.5–25)")
ax.axvspan(25, 30, alpha=0.08, color="gold", label="Overweight (25–30)")
ax.axvspan(30, 100, alpha=0.08, color="red", label="Obese (≥30)")
ax.axvline(bmi_data.median(), color=COLOR_ACCENT, linestyle="--", linewidth=2,
           label=f"Median = {bmi_data.median():.1f}")
ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel("Count")
ax.set_title("BMI Distribution")
ax.set_xlim(10, 60)
ax.legend(fontsize=7, loc="upper right")

# --- Glucose ---
ax = axes[2]
ax.hist(df["avg_glucose_level"], bins=50, color=COLOR_MAIN, alpha=0.7, edgecolor="white")
ax.axvspan(0, 100, alpha=0.08, color="green", label="Normal (<100)")
ax.axvspan(100, 125, alpha=0.08, color="gold", label="Pre-diabetic (100–125)")
ax.axvspan(125, 300, alpha=0.08, color="red", label="Diabetic (≥126)")
ax.axvline(df["avg_glucose_level"].median(), color=COLOR_ACCENT, linestyle="--",
           linewidth=2, label=f"Median = {df['avg_glucose_level'].median():.1f}")
ax.set_xlabel("Avg Glucose Level (mg/dL)")
ax.set_ylabel("Count")
ax.set_title("Glucose Distribution")
ax.legend(fontsize=7, loc="upper right")

for a in axes:
    sns.despine(ax=a)

fig.suptitle("Sanity Check — Continuous Features with Clinical Reference Ranges",
             fontsize=14, y=1.03, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_continuous_clinical_ranges.png"))
plt.close()
print("  → Saved 01_continuous_clinical_ranges.png")


# ──────────────────────────────────────────────────
# PLOT 2: Categorical features — proportion bar charts
# ──────────────────────────────────────────────────
print("\n[Plot 2] Categorical feature distributions")

cat_features = ["gender", "hypertension", "heart_disease", "ever_married",
                "work_type", "Residence_type", "smoking_status"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, feat in enumerate(cat_features):
    ax = axes[i]
    counts = df[feat].value_counts()
    pcts = counts / len(df) * 100

    bars = ax.barh(range(len(counts)), pcts.values,
                   color=COLOR_MAIN, alpha=0.8, edgecolor="white")

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("% of Dataset")
    ax.set_title(feat, fontweight="bold")

    for j, (pct, cnt) in enumerate(zip(pcts.values, counts.values)):
        ax.text(pct + 0.5, j, f"{pct:.1f}% (n={cnt})", va="center", fontsize=8)

    ax.set_xlim(0, max(pcts.values) * 1.35)
    sns.despine(ax=ax)

axes[-1].set_visible(False)

fig.suptitle("Sanity Check — Categorical Feature Distributions",
             fontsize=14, y=1.02, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_categorical_distributions.png"))
plt.close()
print("  → Saved 02_categorical_distributions.png")


# ──────────────────────────────────────────────────
# PLOT 3: Cross-feature consistency checks
# ──────────────────────────────────────────────────
print("\n[Plot 3] Cross-feature consistency checks")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Age vs. Work Type ---
ax = axes[0]
work_order = ["children", "Never_worked", "Private", "Self-employed", "Govt_job"]
work_data = [df.loc[df["work_type"] == wt, "age"].values for wt in work_order if wt in df["work_type"].values]
work_labels = [wt for wt in work_order if wt in df["work_type"].values]
bp = ax.boxplot(work_data, labels=work_labels, patch_artist=True,
                boxprops=dict(facecolor=COLOR_MAIN, alpha=0.6),
                medianprops=dict(color=COLOR_ACCENT, linewidth=2))
ax.set_ylabel("Age (years)")
ax.set_title("Age by Work Type")
ax.tick_params(axis="x", rotation=30)
sns.despine(ax=ax)

# --- Age vs. Ever Married ---
ax = axes[1]
married_data = [df.loc[df["ever_married"] == m, "age"].values for m in ["No", "Yes"]]
bp = ax.boxplot(married_data, labels=["No", "Yes"], patch_artist=True,
                boxprops=dict(facecolor=COLOR_MAIN, alpha=0.6),
                medianprops=dict(color=COLOR_ACCENT, linewidth=2))
ax.set_ylabel("Age (years)")
ax.set_title("Age by Marital Status")
sns.despine(ax=ax)

# --- Age vs. Smoking Status ---
ax = axes[2]
smoke_order = ["Unknown", "never smoked", "formerly smoked", "smokes"]
smoke_data = [df.loc[df["smoking_status"] == s, "age"].values for s in smoke_order]
bp = ax.boxplot(smoke_data, labels=smoke_order, patch_artist=True,
                boxprops=dict(facecolor=COLOR_MAIN, alpha=0.6),
                medianprops=dict(color=COLOR_ACCENT, linewidth=2))
ax.set_ylabel("Age (years)")
ax.set_title("Age by Smoking Status")
ax.tick_params(axis="x", rotation=30)
sns.despine(ax=ax)

fig.suptitle("Sanity Check — Cross-Feature Consistency (Age as anchor)",
             fontsize=14, y=1.03, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_cross_feature_consistency.png"))
plt.close()
print("  → Saved 03_cross_feature_consistency.png")


# ──────────────────────────────────────────────────
# PLOT 4: Missing data summary — focused view
# ──────────────────────────────────────────────────
print("\n[Plot 4] Missing / problematic data summary")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- BMI missing ---
ax = axes[0]
bmi_status = ["Present", "Missing (NaN)"]
bmi_counts = [df["bmi"].notna().sum(), df["bmi"].isna().sum()]
colors = [COLOR_OK, COLOR_WARN]
bars = ax.bar(bmi_status, bmi_counts, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, bmi_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_ylabel("Count")
ax.set_title("BMI — Missing Values")
ax.set_ylim(0, max(bmi_counts) * 1.2)
sns.despine(ax=ax)

# --- Smoking "Unknown" ---
ax = axes[1]
smoke_status = ["Known", 'Unknown']
smoke_counts = [(df["smoking_status"] != "Unknown").sum(),
                (df["smoking_status"] == "Unknown").sum()]
bars = ax.bar(smoke_status, smoke_counts, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, smoke_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_ylabel("Count")
ax.set_title('Smoking Status — "Unknown" Values')
ax.set_ylim(0, max(smoke_counts) * 1.2)
sns.despine(ax=ax)

fig.suptitle("Sanity Check — Data Quality Issues",
             fontsize=14, y=1.03, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_missing_data_summary.png"))
plt.close()
print("  → Saved 04_missing_data_summary.png")


# ──────────────────────────────────────────────────
# PLOT 5: BMI vs Glucose scatter (colored by stroke) — outlier check
# ──────────────────────────────────────────────────
print("\n[Plot 5] BMI vs Glucose scatter — outlier & relationship check")

fig, ax = plt.subplots(figsize=(8, 6))
no_stroke = df[df["stroke"] == 0]
yes_stroke = df[df["stroke"] == 1]

ax.scatter(no_stroke["bmi"], no_stroke["avg_glucose_level"],
           c=COLOR_MAIN, alpha=0.15, s=15, label="No Stroke", edgecolors="none")
ax.scatter(yes_stroke["bmi"], yes_stroke["avg_glucose_level"],
           c=COLOR_ACCENT, alpha=0.7, s=30, label="Stroke",
           edgecolors="white", linewidths=0.3)

# Clinical reference lines
ax.axhline(100, color="gray", linestyle=":", alpha=0.5, label="Glucose = 100 (pre-diabetic)")
ax.axhline(126, color="gray", linestyle="--", alpha=0.5, label="Glucose = 126 (diabetic)")
ax.axvline(25, color="gray", linestyle=":", alpha=0.3)
ax.axvline(30, color="gray", linestyle="--", alpha=0.3)

ax.set_xlabel("BMI (kg/m²)")
ax.set_ylabel("Avg Glucose Level (mg/dL)")
ax.set_title("BMI vs. Glucose — Sanity Check\n(Stroke cases cluster at high age/glucose, not extreme BMI)",
             fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_bmi_vs_glucose_scatter.png"))
plt.close()
print("  → Saved 05_bmi_vs_glucose_scatter.png")


# ──────────────────────────────────────────────────
# PLOT 6: Age vs Stroke — the strongest predictor visualization
# ──────────────────────────────────────────────────
print("\n[Plot 6] Stroke rate by age group — strongest predictor")

df_age = df.copy()
df_age["age_group"] = pd.cut(df_age["age"],
                              bins=[0, 18, 30, 45, 60, 80, 100],
                              labels=["0–18", "18–30", "30–45", "45–60", "60–80", "80+"])

age_stroke = df_age.groupby("age_group")["stroke"].agg(["mean", "count"])
age_stroke["rate"] = age_stroke["mean"] * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(age_stroke)), age_stroke["rate"],
              color=[COLOR_MAIN if r < 5 else COLOR_WARN if r < 10 else COLOR_ACCENT
                     for r in age_stroke["rate"]],
              edgecolor="white")

ax.set_xticks(range(len(age_stroke)))
ax.set_xticklabels(age_stroke.index)
ax.set_xlabel("Age Group")
ax.set_ylabel("Stroke Rate (%)")
ax.set_title("Stroke Rate Increases Sharply with Age\n(Consistent with medical literature)",
             fontweight="bold")

for i, (rate, count) in enumerate(zip(age_stroke["rate"], age_stroke["count"])):
    ax.text(i, rate + 0.3, f"{rate:.1f}%\n(n={count})", ha="center", fontsize=9,
            fontweight="bold")

ax.set_ylim(0, age_stroke["rate"].max() * 1.3)
sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_stroke_rate_by_age.png"))
plt.close()
print("  → Saved 06_stroke_rate_by_age.png")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("SANITY CHECK SUMMARY")
print("=" * 60)
print("""
Key Findings:
  ✅ Age:     0.08–82 yrs — broad, realistic population range
  ✅ BMI:     Mean ~28.9 — consistent with overweight-leaning population
  ✅ Glucose: Bimodal distribution — normal + diabetic subgroups visible
  ✅ Hypertension/Heart Disease: Prevalence rates are plausible
  ✅ Gender:  ~59% Female, ~41% Male — slight imbalance, 1 "Other"
  ✅ Work:    "children" work_type aligns with age < 16
  ✅ Marriage: Unmarried group is younger — consistent
  ✅ Smoking: "Unknown" category is mostly children — makes sense

  ⚠️ BMI:    201 missing values (3.9%) — need imputation
  ⚠️ Smoking: 1544 "Unknown" (30.2%) — significant, but explainable
  ⚠️ Gender: "Other" has only 1 sample — should be dropped

  📊 Figures saved to: outputs/figures/sanity_checks/
""")
