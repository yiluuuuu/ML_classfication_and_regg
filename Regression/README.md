# Project B: Student Exam Score Prediction Using Linear Regression

**Academic Coursework:** Machine Learning Assignment 1  
**Student Name:** Yilkal Bewuketu  
**Student ID:** GSR/2321/19  
**Problem Domain:** Supervised Continuous Regression  
**Target Variable:** Final Mathematics Grade $G3$ ($0 - 20$ scale)  
**Algorithm:** Ordinary Least Squares (OLS) Linear Regression  
**Python Version:** Python 3.10+ (Tested on Python 3.14)

---

## 1. Project Overview

This project implements an end-to-end Machine Learning regression pipeline to predict students' final mathematics exam grades ($G3$) using socio-academic, behavioral, and demographic features from the UCI Student Performance dataset.

### Core Methodological Principles:
1. **Target:** $G3$ (Final Mathematics Grade, scale $0 - 20$).
2. **Predictor Governance:** $G1$ (Period 1 Grade) and $G2$ (Period 2 Grade) are strictly **excluded** from predictors to prevent trivial data leakage and keep the focus on study habits, behavioral attributes, and educational background.
3. **Exploratory Data Analysis:** Thorough univariate, bivariate, and multivariate analysis analyzing distributions, correlations, and interactions.
4. **Data Partitioning:** Leak-free 60/20/20 train/validation/test split ($237$ Train, $79$ Validation, $79$ Test).
5. **Validation-Guided Model Selection:** Compared 3 candidate predictor configurations on the validation set, confirming that combining study time with past academic history and educational aspirations outperforms study time alone.
6. **Unseen Test Evaluation:** Evaluated on the held-out test set using MAE, MSE, RMSE, and $R^2$.
7. **Diagnostics:** Residual normality, homoscedasticity analysis, and coefficient interpretation.
8. **Serialization & Inference:** Serialized with `joblib` and provided standalone inference tools.

---

## 2. Directory Structure

```text
Regression/
├── data/                       # Ingested UCI dataset files
│   ├── student-mat.csv
│   └── student.txt
├── figures/                    # 10 analytical figures
│   ├── 01_target_distribution_g3.png
│   ├── 02_univariate_predictors_hist_box.png
│   ├── 03_bivariate_studytime_vs_g3.png
│   ├── 04_bivariate_failures_and_medu_vs_g3.png
│   ├── 05_multivariate_correlation_heatmap.png
│   ├── 06_multivariate_pairplot.png
│   ├── 07_model_comparison_validation.png
│   ├── 08_predicted_vs_actual_test.png
│   ├── 09_residual_plot_test.png
│   └── 10_regression_coefficients.png
├── models/                     # Serialized model artifact
│   └── linear_regression_student.joblib
├── regression.py               # Complete executable Python pipeline
├── report.md                   # Full 24-section university academic report
├── requirements.txt            # Python dependencies
└── README.md                   # Execution instructions and overview
```

---

## 3. Installation and Setup

### Step A: Navigate to the Regression Directory
```bash
cd /home/yilu/Documents/ML_Assignment/Regression
```

### Step B: Activate the Virtual Environment
```bash
source ../Classification/.venv/bin/activate
```

### Step C: Install Requirements
```bash
pip install -r requirements.txt
```

---

## 4. Execution Instructions

To execute the entire regression pipeline, run:

```bash
python regression.py
```

### What the script executes:
1. **Automated Ingestion:** Loads `student-mat.csv` from `data/` (or downloads it automatically from UCI if missing).
2. **Data Exploration:** Reports descriptive statistics, checks missing values (0), and identifies the 38 zero-grade students (9.6%).
3. **Univariate Analysis:** Plots target $G3$ distribution and key predictors (`figures/01-02`).
4. **Bivariate Analysis:** Analyzes study time, past failures, and parental education vs. $G3$ (`figures/03-04`).
5. **Multivariate Analysis:** Generates correlation heatmaps and pair plots (`figures/05-06`).
6. **Data Partitioning:** Splits data into 60% Train (237), 20% Validation (79), and 20% Test (79).
7. **Validation-Guided Selection:** Evaluates 3 predictor configurations (`figures/07`) and selects Config 3 (Expanded Socio-Academic Profile).
8. **Final Evaluation:** Evaluates on unseen Test data, plotting Predicted vs. Actual (`figures/08`), Residuals (`figures/09`), and Feature Weights (`figures/10`).
9. **Model Persistence:** Serializes model bundle to `models/linear_regression_student.joblib`.
10. **Inference Demonstration:** Demonstrates grade predictions on realistic student profiles.

---

## 5. Summary of Experimental Results

- **Selected Predictor Configuration:** `['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'higher_encoded']`
- **Validation RMSE:** $4.1490$ (improved from $4.2707$ for study time alone)
- **Validation $R^2$:** $0.0548$
- **Test Set Performance (Completely Unseen Data):**
  - **Mean Absolute Error (MAE):** **$3.3894$ grade points**
  - **Mean Squared Error (MSE):** **$19.6402$**
  - **Root Mean Squared Error (RMSE):** **$4.4317$ grade points**
  - **$R^2$ Score:** **$0.2155$ ($21.55\%$ of final grade variance explained)**
- **Fitted Regression Formula:**
  $$\hat{G3} = 9.2403 - 2.0051(\text{failures}) + 0.7998(\text{higher}) + 0.5859(\text{Medu}) - 0.3976(\text{Fedu}) + 0.2030(\text{studytime}) + 0.0210(\text{absences})$$

---

## 6. How to Use the Saved Model for New Predictions

```python
import joblib
import pandas as pd

bundle = joblib.load("models/linear_regression_student.joblib")
model = bundle['model']
features = bundle['features']

# Input student profile
new_student = pd.DataFrame([{
    'studytime': 3,         # 5 to 10 hours weekly
    'failures': 0,          # Zero past course failures
    'absences': 2,          # 2 school absences
    'Medu': 4,              # Mother has higher education
    'Fedu': 4,              # Father has higher education
    'higher_encoded': 1     # Plans to pursue higher education
}])[features]

predicted_g3 = model.predict(new_student)[0]
print(f"Predicted Final Grade: {predicted_g3:.2f} / 20")
```

