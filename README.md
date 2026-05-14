# 🚗⚽ ML Project — Car Price & FIFA Player Value Prediction

A full machine learning pipeline applied on two real-world datasets, covering EDA, preprocessing, regression, classification, cross-validation, and ensemble learning.

---

## 📦 Datasets

| Dataset | Target | Task |
|---------|--------|------|
| Car Price | Car price (£) | Regression + Classification |
| FIFA Players | Player market value (M$) + Performance tier | Regression + Classification |

---


## 🚗 Part 1 — Car Price Dataset

### EDA Highlights
- Price distribution is right-skewed
- Strongest predictors: `Year` (positive), `Mileage` (negative), `Engine Size` (positive)
- Correlation heatmap used to identify feature relationships

### Preprocessing
| Step | Method |
|------|--------|
| Train/Test Split | 80/20 (before imputation to prevent leakage) |
| Missing Values | Median for numerical, Mode for categorical |
| Outlier Handling | IQR-based capping |
| Encoding | One-Hot Encoding → 161 features |
| Scaling | StandardScaler |

### Modeling

**Regression — Predicting Exact Price**
- Baseline: Linear Regression → R² = 0.47

**Classification — Price Tiers (Low / Mid / High)**

| Model | Accuracy |
|-------|----------|
| Baseline KNN (k=3) | 86.1% |
| Tuned KNN (Grid Search) | **88.5%** |

- 5-Fold Cross-Validation used for tuning
- Feature scaling confirmed to significantly boost KNN accuracy

---

## ⚽ Part 2 — FIFA Players Dataset

### EDA Highlights
- Target `Value Per M$` is heavily right-skewed (skewness = 7.98) → log transformation applied
- Strongest predictors: `Overall_Rating` (0.56), `Future Potential` (0.50)
- Average rating analyzed per position

### Preprocessing
| Step | Method |
|------|--------|
| Removed zero-value rows | Players with Value = 0 dropped |
| Train/Test Split | 80/20 |
| Outlier Handling | IQR-based capping |
| Encoding | One-Hot + Target Encoding for `Team` |
| Scaling | StandardScaler |

### Modeling

**Regression — Predicting Player Market Value**

| Model | Test R² |
|-------|---------|
| Linear Regression (Degree 1) | 0.4658 |
| Polynomial Regression (Degree 2) | 0.7640 |
| Polynomial Regression (Degree 3) | 0.8876 |
| Polynomial Regression (Degree 4) | **0.9231** |
| Stacking Regressor (RF + KNN → XGBoost) | **0.9243** |

**Classification — Performance Tier (Low / Mid / High / Elite)**

| Model | Accuracy |
|-------|----------|
| Gaussian Naïve Bayes (Baseline) | 70.89% |
| XGBoost Classifier | **85.33%** |

---

## 🔁 Cross-Validation & Stability

5-Fold Cross-Validation applied across all models.

| Model | Mean Score | Std Dev | Stability |
|-------|------------|---------|-----------|
| KNN Classifier | 85.8% | < 0.02 | ✅ Very Stable |
| XGBoost Classifier | 85.9% | 0.0089 | ✅ Very Stable |
| Stacking Regressor | R² 0.8746 | 0.0334 | ✅ Moderately Stable |

95% Confidence Intervals computed for all final models.

---

## 🏆 Baseline vs Advanced System

| Task | Baseline | Advanced | Improvement |
|------|----------|----------|-------------|
| Car Classification | 86.1% | 88.5% | +2.4% |
| Player Regression R² | 0.4658 | 0.9243 | +45.85% |
| Player Classification | 70.89% | 85.33% | +14.4% |
| Generalization Gap | 0.0425 | 0.015 | Much reduced |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn
