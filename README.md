# 🧠 Cost-Sensitive and Interpretable Stroke Risk Prediction on Tabular Clinical Data

> **CS 464 — Introduction to Machine Learning (Spring 2026)**
> Section 3 / Group 24

| Name | ID | Email |
|---|---|---|
| Nevzat Han Altın | 22202085 | han.altin@ug.bilkent.edu.tr |
| Ahmet Büyükberber | 22202896 | ahmet.buyukberber@ug.bilkent.edu.tr |
| Veli Karakaya | 22202039 | veli.karakaya@ug.bilkent.edu.tr |
| Ece Şeşen | 22201637 | ece.sesen@ug.bilkent.edu.tr |
| Umut Çağın Özcan | 22203549 | cagin.ozcan@ug.bilkent.edu.tr |

---

## 📋 Problem Statement

Given patient demographic, clinical, and lifestyle variables, **predict whether the patient is at risk of stroke**. Since stroke is rare (~4.9 % prevalence in this dataset), trivial classifiers achieve deceptively high accuracy by always predicting "no stroke"; therefore, meaningful progress requires **optimizing minority-class detection** and producing **well-calibrated risk scores**.

## 📊 Dataset

[Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — `healthcare-dataset-stroke-data.csv`

| Stat | Value |
|---|---|
| Rows | 5 110 |
| Features | 11 clinical + 1 target |
| Positive class (stroke = 1) | 249 (4.87 %) |
| Negative class (stroke = 0) | 4 861 (95.13 %) |

**Features:** `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
**Dropped:** `id` (identifier only)

---

## 🗂️ Repository Structure

```
cs464-project/
├── README.md                ← You are here (project attack plan)
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── code/ (not implemented yet, overall structure should look like below)
│   ├── 01_eda.py            ← Exploratory Data Analysis & visualization
│   ├── 02_preprocessing.py  ← Imputation, encoding, scaling, splits
│   ├── 03_models.py         ← Model training, imbalance handling, tuning
│   ├── 04_evaluation.py     ← Metrics, calibration, threshold analysis
│   └── 05_interpretability.py ← SHAP, permutation importance, subgroup analysis
└── outputs/
    ├── figures/             ← All plots and visualizations
    └── results/             ← Metric tables, saved models, logs
```

---

## 🚀 Project Attack Plan

### Phase 1 — Exploratory Data Analysis & Visualization

> **Goal:** Understand the data deeply before modeling.

- [ ] **Univariate analysis** — Distributions of every feature (histograms, bar charts, KDEs)
- [ ] **Bivariate analysis** — Feature vs. target (`stroke`) comparisons (box plots, violin plots, grouped bar charts)
- [ ] **Correlation analysis** — Correlation heatmap (numeric features); Cramér's V or chi-squared for categoricals
- [ ] **Class imbalance visualization** — Bar chart showing 249 vs 4 861 split
- [ ] **Missing value audit** — Identify & visualize missingness pattern (primarily `bmi`)
- [ ] **Statistical tests**
  - Chi-squared / Fisher's exact test for categorical features vs. `stroke`
  - Mann–Whitney U / t-test for continuous features vs. `stroke`
  - Point-biserial correlation for binary features vs. `stroke`
- [ ] **Outlier detection** — Box plots & IQR analysis for `avg_glucose_level`, `bmi`, `age`
- [ ] **PCA / Dimensionality Reduction**
  - PCA on scaled numeric features — scree plot & 2D scatter colored by `stroke`
  - t-SNE / UMAP visualization (if PCA doesn't separate well)
- [ ] **Feature Engineering (exploratory)**
  - Age bins (young / middle-aged / senior)
  - BMI categories (underweight / normal / overweight / obese)
  - Glucose categories (normal / pre-diabetic / diabetic)
  - Interaction features (e.g., `age × hypertension`, `age × avg_glucose_level`)
  - Risk-factor count (sum of binary risk indicators)
- [ ] **Summary statistics table** — Mean, median, std, min, max per class

---

### Phase 2 — Preprocessing Pipeline

> **Goal:** Build a clean, reproducible preprocessing pipeline.

- [ ] **Train / Validation / Test split** — 70 / 15 / 15 with stratification on `stroke`
- [ ] **Missing value imputation** — Compare three strategies:
  - Median imputation
  - KNN imputation
  - Iterative (model-based) imputation
  - Optionally add `bmi_missing` binary indicator
- [ ] **Categorical encoding**
  - One-hot encoding (primary)
  - Target encoding (ablation comparison)
- [ ] **Numerical scaling** — StandardScaler (required for LR & MLP)
- [ ] **Pipeline object** — Wrap everything in `sklearn.Pipeline` for reproducibility

---

### Phase 3 — Model Training & Imbalance Handling

> **Goal:** Train models with systematic imbalance-handling ablations.

#### Models

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | Logistic Regression | ML — linear baseline | Interpretable, class-weighted |
| 2 | Random Forest | ML — ensemble | Nonlinear, robust baseline |
| 3 | Gradient Boosting (XGBoost / LightGBM / CatBoost) | ML — boosting | State-of-the-art tabular |
| 4 | MLP with Categorical Embeddings | DL | PyTorch / TensorFlow |
| 5 | *(stretch)* TabTransformer or FT-Transformer | DL | Transformer-based tabular |

#### Imbalance Strategies (ablation grid)

| Strategy | Applicable to |
|---|---|
| `class_weight='balanced'` / `scale_pos_weight` | All sklearn / XGBoost |
| SMOTE | All (applied on training set only) |
| Random oversampling | All |
| Random undersampling | All |
| Focal loss | MLP / DL models |

Each **model × imbalance strategy** combination will be evaluated.

---

### Phase 4 — Hyperparameter Tuning

> **Goal:** Optimize each model using the right metric.

- [ ] **Search method:** RandomizedSearchCV → Optuna (Bayesian) for top models
- [ ] **Primary metric:** AUC-PR (handles imbalance better than AUC-ROC)
- [ ] **Secondary metrics:** F1 (minority class), recall at fixed precision
- [ ] **Cross-validation:** Stratified 5-fold on training set

---

### Phase 5 — Evaluation & Calibration

> **Goal:** Rigorously evaluate and calibrate models.

- [ ] **Threshold tuning**
  - Precision–Recall curve → select threshold for screening scenario (high recall)
  - F1-optimal threshold
  - Cost-sensitive threshold (assign cost to FN vs FP)
- [ ] **Metrics to report**
  - AUC-ROC, AUC-PR
  - Precision, Recall, F1 (minority class)
  - Specificity, NPV
  - Brier score
  - Confusion matrix (at multiple thresholds)
- [ ] **Probability calibration**
  - Platt scaling (sigmoid)
  - Isotonic regression
  - Calibration curves (reliability diagrams) — before & after
- [ ] **Statistical comparison** — McNemar's test or DeLong test for ROC comparison

---

### Phase 6 — Interpretability & Fairness

> **Goal:** Explain model predictions and check for biases.

- [ ] **Global interpretability**
  - Permutation importance (all models)
  - SHAP summary plots (beeswarm + bar)
- [ ] **Local interpretability**
  - SHAP waterfall plots for individual patients
  - LIME explanations (for comparison)
- [ ] **Subgroup / fairness analysis**
  - Performance breakdown by age group, gender
  - Error analysis: which subgroups have highest FN rates?
- [ ] **Logistic Regression coefficients** — odds ratios with confidence intervals

---

### Phase 7 — Report & Deliverables

> **Goal:** Write the final project report.

- [ ] Results comparison table (all models × all strategies)
- [ ] Key visualizations for the report
- [ ] Ablation summary (what helped, what didn't)
- [ ] Discussion: clinical relevance, limitations, future work

---

## 📏 Evaluation Metrics Summary

| Metric | Why |
|---|---|
| **AUC-PR** | Primary ranking metric — robust under class imbalance |
| **AUC-ROC** | Secondary — standard baseline comparison |
| **Recall (sensitivity)** | Critical for stroke screening — minimize missed cases |
| **Precision** | Controls false alarm rate |
| **F1 (minority)** | Harmonic mean of precision & recall for stroke class |
| **Brier Score** | Measures calibration quality |
| **Calibration Curve** | Visual check of predicted vs. actual probabilities |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, evaluation |
| XGBoost / LightGBM / CatBoost | Gradient boosting |
| PyTorch | MLP / deep tabular models |
| imbalanced-learn | SMOTE, oversampling, undersampling |
| SHAP | Model interpretability |
| matplotlib / seaborn | Visualization |
| Optuna | Bayesian hyperparameter tuning |
| scipy | Statistical tests |

---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-org>/cs464-project.git
cd cs464-project

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run EDA
python code/01_eda.py
```

---

## 📌 Key Design Decisions

1. **Stratified splits** — preserve the 4.87 % stroke ratio in train/val/test.
2. **No leakage** — imputation, encoding, and scaling are fit on training data only.
3. **SMOTE only on training fold** — never applied before splitting.
4. **AUC-PR as primary metric** — accuracy is misleading at 95 % majority class.
5. **Multiple thresholds** — report results at F1-optimal, recall-optimized, and cost-optimized thresholds.
6. **Calibration before deployment** — raw probabilities from tree models are often miscalibrated.

---

*Last updated: 2026-03-18*
