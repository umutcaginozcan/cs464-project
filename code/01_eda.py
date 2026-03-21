"""
Phase 1 — Exploratory Data Analysis & Visualization
CS 464 Stroke Prediction Project

Produces all figures and tables needed for Phase 1 of the report/slides.
Run:  python code/01_eda.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
DATA_PATH = os.path.join("data", "healthcare-dataset-stroke-data.csv")
FIG_DIR = os.path.join("outputs", "figures")
RES_DIR = os.path.join("outputs", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

DPI = 300
COLOR_NO = "#3b82f6"   # blue — No Stroke
COLOR_YES = "#ef4444"  # red  — Stroke
PALETTE = {0: COLOR_NO, 1: COLOR_YES}
LABELS = {0: "No Stroke", 1: "Stroke"}

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

CONTINUOUS = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = [
    "gender", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type", "smoking_status",
]

# ────────────────────────────────────────────
# Load & quick clean
# ────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df.drop(columns=["id"], inplace=True)

# Drop the single "Other" gender row
df = df[df["gender"] != "Other"].reset_index(drop=True)

# bmi is read as object because of "N/A" strings — convert
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Stroke=1: {df['stroke'].sum()} ({df['stroke'].mean()*100:.2f}%)")
print(f"Stroke=0: {(df['stroke']==0).sum()} ({(df['stroke']==0).mean()*100:.2f}%)")


# ════════════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ════════════════════════════════════════════
print("\n\n[1/7] Class Distribution")

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["stroke"].value_counts().sort_index()
bars = ax.bar(
    [LABELS[i] for i in counts.index],
    counts.values,
    color=[PALETTE[i] for i in counts.index],
    edgecolor="white",
    width=0.5,
)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
        f"{val}\n({val/len(df)*100:.1f}%)",
        ha="center", va="bottom", fontweight="bold",
    )
ax.set_ylabel("Count")
ax.set_title("Class Distribution — Severe Imbalance")
ax.set_ylim(0, counts.max() * 1.15)
sns.despine()
fig.savefig(os.path.join(FIG_DIR, "01_class_distribution.png"))
plt.close()
print("  → Saved 01_class_distribution.png")


# ════════════════════════════════════════════
# 2. SUMMARY STATISTICS PER CLASS
# ════════════════════════════════════════════
print("\n[1b] Summary Statistics")

summary = df.groupby("stroke")[CONTINUOUS].describe().T
summary.to_csv(os.path.join(RES_DIR, "summary_stats.csv"))
print("  → Saved summary_stats.csv")


# ════════════════════════════════════════════
# 3. BIVARIATE: CONTINUOUS FEATURES (KDE)
# ════════════════════════════════════════════
print("\n[2/7] Continuous Feature Distributions (KDE by class)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, CONTINUOUS):
    for cls in [0, 1]:
        subset = df.loc[df["stroke"] == cls, feat].dropna()
        ax.hist(
            subset, bins=40, density=True, alpha=0.3,
            color=PALETTE[cls], label=LABELS[cls],
        )
        subset.plot.kde(ax=ax, color=PALETTE[cls], linewidth=2)
    ax.set_title(feat)
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
sns.despine()
fig.suptitle("Continuous Features — Stroke vs. Non-Stroke", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_continuous_kde.png"))
plt.close()
print("  → Saved 02_continuous_kde.png")


# ════════════════════════════════════════════
# 4. BIVARIATE: CATEGORICAL FEATURES (STROKE RATE)
# ════════════════════════════════════════════
print("\n[3/7] Categorical Feature Stroke Rates")

n_cats = len(CATEGORICAL)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, feat in enumerate(CATEGORICAL):
    ax = axes[i]
    ct = df.groupby(feat)["stroke"].agg(["mean", "count"])
    ct = ct.sort_values("mean", ascending=False)
    bars = ax.bar(
        range(len(ct)), ct["mean"] * 100,
        color=COLOR_YES, alpha=0.8, edgecolor="white",
    )
    ax.set_xticks(range(len(ct)))
    ax.set_xticklabels(ct.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Stroke Rate (%)")
    ax.set_title(feat)
    # annotate counts
    for j, (rate, count) in enumerate(zip(ct["mean"], ct["count"])):
        ax.text(j, rate * 100 + 0.3, f"n={count}", ha="center", fontsize=7)
sns.despine()

# hide unused subplot
for j in range(n_cats, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Stroke Rate (%) by Categorical Feature", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_categorical_stroke_rates.png"))
plt.close()
print("  → Saved 03_categorical_stroke_rates.png")


# ════════════════════════════════════════════
# 5. MISSING DATA AUDIT
# ════════════════════════════════════════════
print("\n[4/7] Missing Data Audit")

# --- Compute missing data stats ---
missing_audit = []

# Overall missing counts per column
for col in df.columns:
    n_null = df[col].isnull().sum()
    n_unknown = (df[col] == "Unknown").sum() if df[col].dtype == "object" else 0
    total_problem = n_null + n_unknown
    if total_problem > 0:
        missing_audit.append({
            "Feature": col,
            "NaN Count": n_null,
            "Unknown/NA Count": n_unknown,
            "Total Problematic": total_problem,
            "% of Dataset": f"{total_problem / len(df) * 100:.1f}%",
        })

# Add rows for features with zero missing
for col in df.columns:
    if col not in [m["Feature"] for m in missing_audit]:
        missing_audit.append({
            "Feature": col,
            "NaN Count": 0,
            "Unknown/NA Count": 0,
            "Total Problematic": 0,
            "% of Dataset": "0.0%",
        })

missing_df = pd.DataFrame(missing_audit).sort_values("Total Problematic", ascending=False)
missing_df.to_csv(os.path.join(RES_DIR, "missing_data_audit.csv"), index=False)
print(missing_df.to_string(index=False))
print(f"  → Saved missing_data_audit.csv")

# --- BMI missingness vs stroke (key finding) ---
bmi_present = df[df["bmi"].notna()]
bmi_missing_rows = df[df["bmi"].isna()]
rate_present = bmi_present["stroke"].mean() * 100
rate_missing = bmi_missing_rows["stroke"].mean() * 100

bmi_miss_ct = pd.crosstab(df["bmi"].isna(), df["stroke"])
chi2_bmi, p_bmi, _, _ = stats.chi2_contingency(bmi_miss_ct)
print(f"\n  BMI present → stroke rate: {rate_present:.2f}%  (n={len(bmi_present)})")
print(f"  BMI missing → stroke rate: {rate_missing:.2f}%  (n={len(bmi_missing_rows)})")
print(f"  Chi-squared: χ²={chi2_bmi:.3f}, p={p_bmi:.2e}")

# --- smoking_status "Unknown" analysis ---
unknown_smoke = df[df["smoking_status"] == "Unknown"]
frac_under_18 = (unknown_smoke["age"] < 18).mean() * 100
print(f"\n  smoking_status='Unknown': n={len(unknown_smoke)}, "
      f"mean age={unknown_smoke['age'].mean():.1f}, "
      f"median age={unknown_smoke['age'].median():.1f}")
print(f"  Fraction under 18: {frac_under_18:.1f}%")

# --- Visualization: missingno matrix (full dataset view) ---
import missingno as msno

# Treat smoking_status "Unknown" as NaN for the matrix visualization
df_msno = df.copy()
df_msno.loc[df_msno["smoking_status"] == "Unknown", "smoking_status"] = np.nan

fig, ax = plt.subplots(figsize=(12, 8))
msno.matrix(df_msno, ax=ax, fontsize=10, sparkline=False, color=(0.23, 0.51, 0.96))
ax.set_title("Missingness Matrix — Full Dataset (5109 × 11)\nWhite = Missing / Unknown",
             fontsize=13, pad=15)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_missing_data_audit.png"))
plt.close()
print("  → Saved 04_missing_data_audit.png")

# --- Save missingness ↔ stroke correlation to stat_tests.csv later ---
# We'll append these as extra rows when stat_tests are computed below
missingness_tests = []

# bmi_missing vs stroke
n_bmi = len(df)
cramers_v_bmi = np.sqrt(chi2_bmi / n_bmi)
missingness_tests.append({
    "Feature": "bmi_missing (indicator)",
    "Test": "Chi-squared",
    "Statistic": f"χ²={chi2_bmi:.2f}",
    "p-value": f"{p_bmi:.2e}",
    "Effect Size": f"V={cramers_v_bmi:.3f}, stroke rate: {rate_present:.1f}% vs {rate_missing:.1f}%",
    "Significant": "Yes",
})

# smoking_unknown vs stroke
smoke_unk_ct = pd.crosstab(df["smoking_status"] == "Unknown", df["stroke"])
chi2_smoke, p_smoke, _, _ = stats.chi2_contingency(smoke_unk_ct)
cramers_v_smoke = np.sqrt(chi2_smoke / n_bmi)
rate_known = df.loc[df["smoking_status"] != "Unknown", "stroke"].mean() * 100
rate_unk = df.loc[df["smoking_status"] == "Unknown", "stroke"].mean() * 100
missingness_tests.append({
    "Feature": "smoking_unknown (indicator)",
    "Test": "Chi-squared",
    "Statistic": f"χ²={chi2_smoke:.2f}",
    "p-value": f"{p_smoke:.2e}",
    "Effect Size": f"V={cramers_v_smoke:.3f}, stroke rate: {rate_known:.1f}% vs {rate_unk:.1f}%",
    "Significant": "Yes" if p_smoke < 0.05 else "No",
})


# ════════════════════════════════════════════
# 6. STATISTICAL TESTS
# ════════════════════════════════════════════
print("\n[5/7] Statistical Tests")

test_results = []

# --- Continuous features: Mann-Whitney U ---
stroke_mask = df["stroke"] == 1
for feat in CONTINUOUS:
    grp0 = df.loc[~stroke_mask, feat].dropna()
    grp1 = df.loc[stroke_mask, feat].dropna()
    u_stat, p_val = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
    # rank-biserial correlation as effect size
    n0, n1 = len(grp0), len(grp1)
    rbc = 1 - (2 * u_stat) / (n0 * n1)
    test_results.append({
        "Feature": feat,
        "Test": "Mann-Whitney U",
        "Statistic": f"U={u_stat:.0f}",
        "p-value": f"{p_val:.2e}",
        "Effect Size": f"r_rb={rbc:.3f}",
        "Significant": "Yes" if p_val < 0.05 else "No",
    })

# --- Categorical features: Chi-squared ---
for feat in CATEGORICAL:
    ct = pd.crosstab(df[feat], df["stroke"])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1))) if (k - 1) > 0 else 0

    # Odds ratio for binary features
    effect_str = f"V={cramers_v:.3f}"
    if ct.shape[0] == 2:
        # 2×2 table — compute odds ratio
        a, b = ct.iloc[0, 0], ct.iloc[0, 1]
        c, d = ct.iloc[1, 0], ct.iloc[1, 1]
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.inf
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d) > 0 else np.inf
        ci_low = np.exp(np.log(odds_ratio) - 1.96 * se_log_or)
        ci_high = np.exp(np.log(odds_ratio) + 1.96 * se_log_or)
        effect_str += f", OR={odds_ratio:.2f} [{ci_low:.2f}–{ci_high:.2f}]"

    test_results.append({
        "Feature": feat,
        "Test": "Chi-squared",
        "Statistic": f"χ²={chi2:.2f}",
        "p-value": f"{p_val:.2e}",
        "Effect Size": effect_str,
        "Significant": "Yes" if p_val < 0.05 else "No",
    })

test_results.extend(missingness_tests)  # append missingness ↔ stroke correlation

test_df = pd.DataFrame(test_results)
test_df.to_csv(os.path.join(RES_DIR, "stat_tests.csv"), index=False)
print(test_df.to_string(index=False))
print("  → Saved stat_tests.csv")


# ════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ════════════════════════════════════════════
print("\n[6/7] Correlation Heatmap")

numeric_cols = CONTINUOUS + ["hypertension", "heart_disease", "stroke"]
corr_df = df[numeric_cols].dropna()
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
)
ax.set_title("Correlation Matrix (Numeric Features + Target)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_correlation_heatmap.png"))
plt.close()
print("  → Saved 05_correlation_heatmap.png")


# ════════════════════════════════════════════
# 8. PCA VISUALIZATION
# ════════════════════════════════════════════
print("\n[7a/7] PCA")

pca_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
pca_df = df[pca_features + ["stroke"]].dropna().copy()

X_pca = pca_df[pca_features].values
y_pca = pca_df["stroke"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA()
X_proj = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scree plot
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_ * 100,
            color="#64748b", edgecolor="white")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("Scree Plot")
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
axes[0].plot(range(1, len(cumvar) + 1), cumvar, "o-", color=COLOR_YES, markersize=5)
axes[0].axhline(y=90, color="gray", linestyle="--", alpha=0.5)

# 2D scatter
for cls in [0, 1]:
    mask = y_pca == cls
    axes[1].scatter(
        X_proj[mask, 0], X_proj[mask, 1],
        c=PALETTE[cls], label=LABELS[cls],
        alpha=0.3 if cls == 0 else 0.7,
        s=10 if cls == 0 else 30,
        edgecolors="none",
    )
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[1].set_title("PCA — Stroke vs. Non-Stroke")
axes[1].legend()

sns.despine()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_pca.png"))
plt.close()
print("  → Saved 06_pca.png")

# ════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — All outputs saved.")
print(f"  Figures: {FIG_DIR}/")
print(f"  Tables:  {RES_DIR}/")
print("=" * 60)
