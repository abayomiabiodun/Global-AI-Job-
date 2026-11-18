# Global-AI-Job

# Global AI Jobs 2026: Salary Prediction

An end-to-end, leakage-safe machine-learning project that predicts the **salary (USD) of an AI job posting**, identifies what actually drives pay, and returns **calibrated 90% prediction intervals** for negotiation ranges.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-End--to--End-brightgreen)

---

## Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Key Findings](#-key-findings)
- [Recommendations](#-recommendations)
- [Using the Saved Model](#-using-the-saved-model)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Author](#-author)

---

## Problem Statement

AI roles are among the fastest-growing and most unevenly paid jobs in the world. Compensation teams struggle to benchmark offers, candidates can't tell whether an offer is fair, and analysts lack a clear view of where pay gaps come from.

**Business question:** *Given the attributes of an AI job posting, what salary should we expect, and which attributes actually drive it?*

**ML formulation:** supervised regression on `Salary_USD` (right-skewed, so modelled as `ln(salary)`). One row = one job posting.

| Stakeholder | Use |
|---|---|
| HR / compensation teams | Benchmark a new role's salary band before publishing |
| Candidates | Sanity-check an offer against the market |
| Analysts / policy makers | Quantify the pay premium of seniority, geography and role |

**Success criteria (set before modelling)**
- Test R² on log-salary clearly above the baseline, with a train–CV–test gap < 0.03
- Errors in business terms: MAE in USD and median absolute percentage error
- **Zero target leakage**
- 90% prediction intervals that really cover ≈ 90% of unseen postings

---

## Dataset

- **File:** `global_ai_jobs_2026.csv`
- **Size:** 5,000 job postings × 37 columns

**Cleaning and audit highlights**
- Blank certifications recoded to "None"
- Constants, IDs and synthetic future dates removed (postings dated as far out as 2039)
- Logical-consistency and outlier checks

**Target-leakage audit**
`Salary_Min`, `Salary_Max` and `Salary_Category` are all derived from the target (min ≈ 0.7 to 0.9 × salary, max ≈ 1.1 to 1.4 × salary, plus fixed salary bins). Keeping them would have inflated R² from ≈ 0.78 to ≈ 0.99, so they were removed. Features that only exist *after* posting or hiring (applicant counts, satisfaction scores) were tested in an ablation study and excluded from the deployable model.

---

## Approach

1. **Audit & clean**, then the **leakage audit**
2. **Split first** (80/20, stratified). EDA is done on training rows only.
3. **EDA & statistical tests:** η², Kruskal–Wallis, Spearman, mutual information, VIF, ANOVA interaction test, skill-premium tests with FDR control
4. **Target transform:** `ln(salary)` (Box–Cox λ ≈ −0.02)
5. **Feature engineering:** ordinal maps, log/sqrt, ratios, multi-hot skills with IDF rarity, and cross-fitted shrinkage target encoding
6. **Feature reduction:** Marchenko–Pastur / PCA analysis, permutation importance, Lasso and an ablation study with the one-standard-error rule (28 raw features reduced to **2**)
7. **Scaler comparison:** Standard / Robust / MinMax / Yeo-Johnson
8. **Modelling:** dummy baseline, Ridge, Lasso, ElasticNet, KNN, Random Forest, ExtraTrees, HistGB, XGBoost and LightGBM, plus tuned boosters and an ensemble. Compared with repeated and nested CV; the final pick uses the one-SE rule.
9. **Diagnostics:** learning and validation curves, bootstrap bias–variance decomposition, noise-ceiling test
10. **Leakage verification:** leaky-column demo, naive-vs-pipeline encoding demo, label-permutation test, adversarial validation, duplicate check
11. **Uncertainty:** Mondrian **conformal prediction intervals** by seniority level, plus SHAP

**Final model:** Ridge + LightGBM ensemble, chosen by the one-SE rule. All top models score within ≈ 0.005 R² of each other, and plain Ridge is practically equivalent.

---

## Key Results

| Metric | Value |
|---|---|
| **Test R² (log salary)** | **0.788** (95% bootstrap CI 0.768 to 0.807) |
| Nested-CV R² (tuned LightGBM) | 0.780 |
| Train − test R² gap | +0.003 |
| Test MAE | $27,893 |
| **Test median absolute % error** | **16.8%** (baseline 29.1%) |
| Typical error (RMSE of log salary) | ≈ 22% |
| 90% interval empirical coverage | **90.2%** |
| Shuffled-label R² | −0.006 (≈ 0, as it should be) |
| Adversarial-validation AUC | 0.507 (≈ 0.5, train and test indistinguishable) |

---

## Key Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Seniority is the dominant driver** (Executive ≈ +371%, Lead ≈ +194%, Senior ≈ +112%, Mid ≈ +49% vs Entry) | η² ≈ 0.65; permutation importance |
| 2 | **Country is the second driver** (at equal seniority, the US, Israel, Singapore and Switzerland pay ≈ +105 to 119% vs India; priciest-to-cheapest spread ≈ ×2.4) | η² ≈ 0.13 |
| 3 | **Nothing else measurably matters:** job title, industry, specialisation, education, skills, company size, remote work, AI investment, equity and bonus % | 0/30 skills significant after FDR; importance ≈ 0 |
| 4 | `Years_of_Experience` is unrelated to salary or to `Experience_Level` | ρ ≈ 0 |
| 5 | Salary is **log-additive** (no seniority × geography interaction) | ANOVA interaction p = 0.53 |
| 6 | Simple linear models match boosted trees | All top models within ≈ 0.005 R² |
| 7 | No under- or over-fitting | Gaps ≈ 0.003 to 0.008; variance ≈ 2% of MSE |
| 8 | No leakage | Five independent tests |

---

## Recommendations

**Compensation / HR teams**
- **Price roles on a seniority × country grid.** Use the model's estimate as the midpoint and the conformal band (≈ ×/÷ 1.4 at 90%) as the negotiation range.
- Don't justify pay differences with title, industry, skills, education or company size using this data. No effect was detected.
- Use the model for **benchmarking and planning**, not for automatically setting an individual's pay.

**Data team**
- Quarantine target-derived fields (`Salary_Min/Max/Category`) so they can never enter a feature matrix
- Add ingestion validation (future-dated postings, `Experience_Level` vs `Years_of_Experience`, ~30% disagreement between `Remote_Work_Pct` and `Work_From_Home_Available`)
- **Validate on real, external postings before production**

**ML engineering**
- Ship the whole pipeline, not just the model. Production inputs only need `Experience_Level` and `Country`.
- Monitor monthly (country/seniority mix, interval coverage, error by segment) and retrain quarterly
- Next experiments: hierarchical / mixed-effects models, country-level covariates (PPP, GDP, cost of living), quantile regression for heteroscedastic bands, time-aware validation once genuine dates exist

---

## Using the Saved Model

The notebook saves the full pipeline to `salary_model.joblib` (feature engineering → encoding → model, plus conformal quantiles). Only `Experience_Level` and `Country` influence predictions.

Example outputs from the notebook:

| Experience level | Country | Estimate | 90% band |
|---|---|---|---|
| Entry Level | India | $50,539 | $35,411 – $72,130 |
| Senior Level | Germany | $193,426 | $139,912 – $267,407 |
| Senior Level | United States | $204,766 | $148,115 – $283,086 |
| Executive | Switzerland | $450,177 | $292,665 – $692,461 |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<my-username>/<my-repo-name>.git
cd <my-repo-name>
```

### 2. Install dependencies
```bash
pip install pandas numpy scipy scikit-learn statsmodels xgboost lightgbm shap matplotlib seaborn joblib jupyter
```

### 3. Add the data
Place `global_ai_jobs_2026.csv` next to the notebook.

### 4. Run the notebook
```bash
jupyter notebook global_ai_jobs_salary_analysis.ipynb
```
Choose **Kernel → Restart & Run All**.

---

## Repository Structure

```
├── global_ai_jobs_salary_analysis.ipynb   # Full end-to-end notebook
├── global_ai_jobs_2026.csv                # Job postings data (add locally)
├── salary_model.joblib                    # Saved pipeline (generated by the notebook)
└── README.md
```

---

## Limitations

- **The dataset appears synthetic** (independent, uniformly distributed numerics and exactly generated salary bands). The conclusion that skills and job title don't matter is a property of *this* data and is unlikely to hold in real labour markets. Don't quote the absolute magnitudes (e.g. +371% for Executives) as real-world facts.
- ≈ 22% of variance is unexplained and apparently irreducible with the supplied columns
- Country effects rely on ≈ 80 training rows per country, so very small markets have wider uncertainty
- Findings are associations, not causal effects
- Dates are unusable, so no temporal validation was possible

---

## Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
