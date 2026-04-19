# 🌍 Global Longevity: A Statistical Deep Dive
### Into Life Expectancy Drivers Across Nations

| | |
|---|---|
| **Author** | Kartik Singh |
| **Date** | April 2026 |
| **Domain** | Global Public Health · Predictive Analytics |
| **Tools** | Python, Pandas, NumPy, Scikit-learn, Statsmodels, XGBoost, SHAP |

---

## 📌 Project Overview

Life expectancy is one of the most comprehensive indicators of a nation's overall well-being. This project conducts a rigorous, end-to-end statistical analysis of global life expectancy data by integrating **five independent datasets** spanning economic development, environmental health, education, and subjective well-being across **150+ countries from 2015 to 2023**.

The analysis follows a structured data science pipeline — from raw data ingestion and quality assurance through to predictive modeling, hypothesis testing, and model evaluation — producing interpretable, statistically validated findings on the primary drivers of longevity worldwide.

**Core Research Question:**
> *Which socioeconomic, environmental, and institutional factors most significantly predict life expectancy across nations, and how accurately can these factors forecast longevity?*

---

## 🎯 Project Objectives

| # | Objective | Methodology |
|---|-----------|-------------|
| 1 | Data Sanitization & Quality Assurance | Missing value imputation, duplicate removal, IQR outlier treatment |
| 2 | Exploratory Data Analysis & Feature Correlation | Distribution analysis, heatmaps, scatter plots, regional comparisons |
| 3 | Predictive Modeling via Linear Regression | OLS regression, feature selection, train/test evaluation |
| 4 | Hypothesis Testing & LINE Assumption Validation | p-value testing, Durbin-Watson, Shapiro-Wilk, homoscedasticity |
| 5 | Model Evaluation & Performance Metrics | R², MAE, RMSE, MAPE, 5-fold cross-validation, regional breakdown |
| ⭐ | Bonus: XGBoost + SHAP Interpretability | Gradient boosting benchmark, Shapley value explainability |

---

## 📁 Repository Structure

```
project/
│
├── LIFE_EXPECTANCY_1.IPYNB         # Main analysis notebook (all 42 steps)
│
├── Data/
│   ├── WHR_2015.csv                # World Happiness Report — 2015
│   ├── WHR_2016.csv                # World Happiness Report — 2016
│   ├── ...                         # (WHR_2017 through WHR_2023)
│   ├── WHR_2023.csv                # World Happiness Report — 2023
│   ├── Global_Life_Expectancy_Historical.csv
│   ├── UN_Human_Development_Index_1990_2022.csv
│   ├── Global_PM25_Air_Pollution_2010_2019.csv
│   ├── World_Bank_Country_Indicators_Data.csv
│   ├── Life_Expectancy_by_Country_Long_Trend.csv
│   │
│   ├── master_longevity.csv        # Auto-generated: merged raw dataset
│   └── master_longevity_clean.csv  # Auto-generated: cleaned analysis dataset
│
└── README.md
```

> **Note:** The two `master_longevity*.csv` files are generated automatically when you run the notebook. You do not need to create them manually.

---

## 🗃️ Datasets

| Dataset | Source | Coverage | Features Extracted |
|---------|--------|----------|--------------------|
| World Happiness Report (2015–2023) | Gallup / WHR | 9 annual files | GDP per capita, social support, freedom, generosity, corruption perceptions, happiness score |
| Global Life Expectancy | Our World in Data | 1950–2023 | `life_expectancy` — the **target variable** |
| UN Human Development Index | UNDP | 1990–2022 | Expected years of schooling, HDI score, GNI per capita |
| Global PM2.5 Air Pollution | WHO / GHDx | 2010–2019 | Fine particulate matter concentration (µg/m³) |
| World Bank Country Indicators | World Bank | 1960–2024 | Health expenditure (% of GDP), GDP per capita |

### Dataset Merge Architecture

The World Happiness Report (2015–2023) serves as the **structural backbone**. All other datasets are joined using `country` and `year` as composite keys via sequential left joins:

```
World Happiness Report (2015–2023)       ← base frame (country + year)
          │
          ▼  left join on [country, year]
Global Life Expectancy Historical         ← adds: life_expectancy (TARGET)
          │
          ▼  left join on [country, year]
UN Human Development Index                ← adds: schooling_years, hdi_score
          │
          ▼  left join on [country, year]
Global PM2.5 Air Pollution                ← adds: pm25 concentration
          │
          ▼  left join on [country, year]
World Bank Indicators                     ← adds: gdp_per_capita_wb, health_expenditure_pct
          │
          ▼
    master_longevity.csv                  ← final analysis-ready dataset
```

**Final dataset:** 1,296 country-year observations · 158 countries · 2015–2023

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Install Dependencies

Run this in your terminal or directly in the first notebook cell:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels xgboost shap
```

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data ingestion, transformation, numerical operations |
| `matplotlib`, `seaborn` | Statistical visualizations and publication-quality charts |
| `scikit-learn` | Linear Regression, train/test split, cross-validation, metrics |
| `scipy`, `statsmodels` | Statistical testing, OLS regression, LINE assumption checks |
| `xgboost` | Gradient-boosted ensemble model for benchmark comparison |
| `shap` | Model interpretability via Shapley Additive Explanations |

### Running the Notebook

1. Place all CSV files inside a folder named `Data/` in the same directory as the notebook.
2. Launch Jupyter:
   ```bash
   jupyter notebook LIFE_EXPECTANCY_1.IPYNB
   ```
3. Run all cells top to bottom (`Kernel → Restart & Run All`).

The notebook will automatically generate `master_longevity.csv` and `master_longevity_clean.csv` inside the `Data/` folder and save all 13 plots as PNG files.

---

## 🔬 Methodology

### Phase 1 — Environment Setup (Steps 1–2)
Installation and import of all required libraries for data processing, visualization, statistical testing, and machine learning.

---

### Phase 2 — Building the Master Dataset (Steps 3–9)
Nine annual WHR files (2015–2023) are stacked into a unified base frame. Five datasets are then merged sequentially using `country` and `year` as keys. A long-trend life expectancy file is used as a gap-filler for missing values in the primary target column. The final merged dataset is saved as `master_longevity.csv`.

---

### Objective 1 — Data Sanitization & Quality Assurance (Steps 10–16)

| Step | Action |
|------|--------|
| 10 | Load master CSV and inspect shape, dtypes, and descriptive statistics |
| 11 | Standardize all column names to lowercase with underscores |
| 12 | Remove duplicate rows |
| 13 | Impute missing numeric values with **median** (robust to skew); fill `region` with **mode** |
| 14 | Cap outliers using the **IQR method** (1.5× IQR boundary) — preserves all 1,296 rows |
| 15 | Label-encode the `region` categorical variable for use in ML models |
| 16 | Save the cleaned dataset as `master_longevity_clean.csv` |

**Missing value treatment summary:**

| Column | Missing % | Treatment |
|--------|-----------|-----------|
| `perceptions_of_corruption` | 0.1% | Median imputation |
| `health_expenditure_pct` | 9.6% | Median imputation |
| `gdp_per_capita_wb` | 9.0% | Median imputation |
| `schooling_years` | 17.6% | Median imputation |
| `hdi_score` | 17.6% | Median imputation |
| `pm25` | 43.4% | Median imputation — flagged as low-coverage |

---

### Objective 2 — Exploratory Data Analysis (Steps 17–22)

Six visualizations are produced to understand distributions, correlations, and regional patterns:

- **Distribution of life expectancy** — slight left skew; most countries cluster between 70–80 years
- **Correlation heatmap** — HDI score (r = +0.86) and healthy life expectancy (r = +0.85) are the strongest positive predictors; PM2.5 (r = −0.32) is the key negative driver
- **Scatter plots (top 4 features)** — confirm linear relationships; GDP per capita shows a slight Preston Curve effect at high incomes
- **Regional box plots & bar charts** — Western Europe exceeds 80-year median; Sub-Saharan Africa falls below 65 years, a 15+ year gap
- **Global trend 2015–2023** — steady improvement pre-COVID, a visible dip in 2020–2021, partial recovery by 2023
- **Feature correlation bar chart** — ranked visualization of all Pearson correlations with the target

---

### Objective 3 — Predictive Modeling (Steps 23–27)

**Feature selection:** Ten predictors are chosen based on correlation analysis. Two features are deliberately excluded:
- `gdp_per_capita_wb` — strongly collinear with `gdp_per_capita` (r = 0.726), would inflate standard errors
- `healthy_life_expectancy` — a WHR wellness index that proxies the target variable directly, creating **data leakage risk**

| Feature | Correlation with Target | Direction |
|---------|------------------------|-----------|
| `hdi_score` | +0.859 | Positive |
| `gdp_per_capita` | +0.796 | Positive |
| `schooling_years` | +0.773 | Positive |
| `happiness_score` | +0.760 | Positive |
| `social_support` | +0.590 | Positive |
| `freedom_to_make_life_choices` | +0.381 | Positive |
| `health_expenditure_pct` | +0.375 | Positive |
| `perceptions_of_corruption` | +0.332 | Positive |
| `pm25` | −0.324 | Negative |
| `region_encoded` | — | Categorical proxy |

An 80/20 train-test split is applied (random_state=42). A scikit-learn `LinearRegression` model is trained and evaluated on the held-out test set.

---

### Objective 4 — Hypothesis Testing & LINE Validation (Steps 28–34)

The full OLS model is re-run via **statsmodels** to obtain p-values, confidence intervals, and the F-statistic.

**Hypotheses for each feature:**
- **H₀:** The feature has no significant effect on life expectancy (β = 0)
- **Hₐ:** The feature has a statistically significant effect (β ≠ 0)
- **Decision rule:** Reject H₀ when p-value < 0.05

**LINE Assumption Checks:**

| Assumption | Test Used | Expected Result |
|------------|-----------|-----------------|
| **L** — Linearity | Scatter plots (Step 19) | Linear trend confirmed visually |
| **I** — Independence | Durbin-Watson test | Acceptable range: 1.5 – 2.5 |
| **N** — Normality | Shapiro-Wilk + Q-Q plot | p > 0.05 or Q-Q points follow the line |
| **E** — Equal Variance | Residuals vs Fitted plot | Horizontal band around zero |

---

### Objective 5 — Model Evaluation (Steps 35–41)

Linear Regression is benchmarked against **XGBoost** (300 estimators, learning rate 0.05, max depth 6) to quantify the benefit of capturing non-linear relationships.

**Evaluation metrics:** R², MAE, RMSE, MAPE

**Stability check:** 5-fold cross-validation on the full dataset for both models

**Regional breakdown:** RMSE computed per world region — Western Europe and North America show lowest error; Sub-Saharan Africa shows highest, reflecting unmodeled conflict and disease burden factors.

**SHAP interpretability:**
- Beeswarm plot — shows direction and magnitude of each feature's contribution across all test countries
- Waterfall plot — explains a single country's prediction step by step
- Bar plot — global average feature importance by SHAP value

---

## 📊 Key Findings

### Strongest Predictors of Life Expectancy

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | HDI Score | +0.859 | Human development is the dominant driver of longevity |
| 2 | GDP per Capita | +0.796 | Economic prosperity enables better healthcare and nutrition |
| 3 | Schooling Years | +0.773 | Education improves health literacy and economic mobility |
| 4 | Happiness Score | +0.760 | Subjective well-being reflects systemic quality of life |
| 5 | PM2.5 Pollution | −0.324 | Higher air pollution consistently lowers life expectancy |

### Regional Disparities
- **Western Europe:** Median life expectancy > 80 years
- **Sub-Saharan Africa:** Median life expectancy < 65 years
- **Gap of 15+ years** driven by differences in healthcare access, income, and disease burden

### COVID-19 Impact
Global life expectancy declined visibly in 2020–2021 — the first sustained drop in decades. Recovery began in 2022–2023, but lower-income nations have not yet returned to their pre-pandemic trajectory.

---

## 📈 Model Performance

| Metric | Linear Regression | XGBoost |
|--------|:-----------------:|:-------:|
| R² Score | ~0.88 | ~0.96+ |
| MAE (years) | < 2.0 | < 1.5 |
| RMSE (years) | < 3.5 | < 2.0 |
| CV Mean R² | Stable | Stable |

> XGBoost significantly outperforms Linear Regression by capturing non-linear threshold effects (e.g., diminishing longevity returns at very high income levels). However, Linear Regression retains value for interpretability — its coefficients directly quantify the marginal effect of each policy lever.

---

## 🖼️ Visualizations Generated

| File | Description |
|------|-------------|
| `plot_01_life_expectancy_distribution.png` | Histogram + KDE of the target variable |
| `plot_02_correlation_heatmap.png` | Full feature correlation matrix |
| `plot_03_scatter_top4.png` | Top 4 features vs life expectancy with trend lines |
| `plot_04_regional_comparison.png` | Box plot + bar chart by world region |
| `plot_05_trend_over_time.png` | Global mean life expectancy trend 2015–2023 |
| `plot_06_correlation_bar.png` | Ranked Pearson correlations with life expectancy |
| `plot_07_lr_performance.png` | Actual vs Predicted + Residuals Distribution |
| `plot_08_lr_coefficients.png` | Linear Regression feature coefficients |
| `plot_09_normality.png` | Q-Q plot + Residuals histogram (normality check) |
| `plot_10_homoscedasticity.png` | Residuals vs Fitted + Scale-Location plot |
| `plot_11_evaluation_dashboard.png` | 4-chart model evaluation dashboard |
| `plot_12_xgb_feature_importance.png` | XGBoost feature importance (gain scores) |
| `plot_13_policy_recommendations.png` | Policy interventions and estimated life expectancy impact |

All plots are saved at **200 DPI** in the `Data/` folder automatically during notebook execution.

---

## 💡 Policy Recommendations

Based on the model's findings and SHAP analysis, the five highest-impact interventions are:

| Policy Lever | Estimated Gain | Priority |
|-------------|----------------|----------|
| Increase HDI Score by 0.1 units | +6.2 years | 🔴 Critical |
| Reduce PM2.5 from 35 → 10 µg/m³ | +2.5 years | 🟠 High |
| Raise health expenditure by +3% GDP | +1.1 years | 🟠 High |
| Increase GDP per capita by 10% | +0.8 years | 🟡 Medium |
| Add 1 year of schooling | +0.16 years | 🟡 Medium |

---

## ⚠️ Limitations

- **PM2.5 data ends at 2019** — missing values for 2020–2023 filled by median imputation
- **WHR coverage** is limited to approximately 158 countries per year — some nations are excluded
- **HDI score is partially collinear** with life expectancy since both reflect overall development level
- **Country name mismatches** across datasets may have excluded some observations during merging
- **Linear Regression assumes linearity** — the Preston Curve effect suggests diminishing returns at high GDP levels

---

## 🔮 Future Directions

- Add infant mortality rates, immunization coverage, and disease burden data (WHO databases)
- Apply **panel regression with country fixed effects** to control for time-invariant country characteristics
- Train **region-specific models** to reduce the elevated RMSE observed for Sub-Saharan Africa and South Asia
- Extend the time horizon back to 2000–2023 using the full historical life expectancy dataset
- Incorporate conflict indices and governance scores as additional predictors

---

## 📋 Project Completion Checklist

| Objective | Status |
|-----------|--------|
| Data Sanitization & Quality Assurance | ✅ Complete |
| Exploratory Data Analysis & Feature Correlation | ✅ Complete |
| Predictive Modeling — Linear Regression | ✅ Complete |
| Hypothesis Testing & LINE Validation | ✅ Complete |
| Model Evaluation — R², MAE, RMSE, Cross-Validation | ✅ Complete |
| Bonus — XGBoost + SHAP Interpretability | ✅ Complete |

---

*Project completed — April 2026*
