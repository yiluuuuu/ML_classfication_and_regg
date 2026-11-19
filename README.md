# Machine Learning Assignment 1
**Coursework Submission:** Machine Learning Assignment 1  
**Student Name:** Yilkal Bewuketu  
**Student ID:** GSR/2321/19  
**Supervisor / Instructor:** Course Instructor  
**Date:** Academic Year 2026  
**Status:** Completed and Verified  

---

## Repository Structure

```text
ML_Assignment/
├── Classification/                     # PROJECT A: Multi-Class Botanical Classification
│   ├── figures/                        # 12 analytical figures
│   │   ├── 01_univariate_histograms.png
│   │   ├── 02_univariate_boxplots.png
│   │   ├── 03_species_distribution.png
│   │   ├── 04_bivariate_sepal_length_vs_width.png
│   │   ├── 05_bivariate_petal_length_vs_width.png
│   │   ├── 06_features_by_species_boxplots.png
│   │   ├── 07_multivariate_pairplot.png
│   │   ├── 08_multivariate_correlation_heatmap.png
│   │   ├── 09_hyperparameter_tuning_depth.png
│   │   ├── 10_feature_importance.png
│   │   ├── 11_decision_tree_structure.png
│   │   └── 12_confusion_matrix.png
│   ├── models/                         # Serialized model artifact
│   │   └── decision_tree_iris.joblib
│   ├── classification.py               # Complete executable Python pipeline
│   ├── report.md                       # Comprehensive 24-section academic report
│   ├── requirements.txt                # Project-specific dependencies
│   └── README.md                       # Project documentation
│
├── Regression/                         # PROJECT B: Student Exam Score Regression
│   ├── data/                           # UCI Student Performance Dataset
│   │   ├── student-mat.csv
│   │   └── student.txt
│   ├── figures/                        # 10 analytical figures
│   │   ├── 01_target_distribution_g3.png
│   │   ├── 02_univariate_predictors_hist_box.png
│   │   ├── 03_bivariate_studytime_vs_g3.png
│   │   ├── 04_bivariate_failures_and_medu_vs_g3.png
│   │   ├── 05_multivariate_correlation_heatmap.png
│   │   ├── 06_multivariate_pairplot.png
│   │   ├── 07_model_comparison_validation.png
│   │   ├── 08_predicted_vs_actual_test.png
│   │   ├── 09_residual_plot_test.png
│   │   └── 10_regression_coefficients.png
│   ├── models/                         # Serialized model artifact
│   │   └── linear_regression_student.joblib
│   ├── regression.py                   # Complete executable Python pipeline
│   ├── report.md                       # Comprehensive 24-section academic report
│   ├── requirements.txt                # Project-specific dependencies
│   └── README.md                       # Project documentation
│
├── app.py                              # Interactive Streamlit Web Application (Model Deployment)
├── requirements.txt                    # Unified dependency file
├── .gitignore                          # Git configuration for deployment
└── README.md                           # Top-level documentation & quick-start guide
```

---

## Quick-Start: How to Execute Both Projects

### 1. Set Up Environment
```bash
cd /home/yilu/Documents/ML_Assignment
python3 -m venv Classification/.venv
source Classification/.venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Project A (Classification)
```bash
cd /home/yilu/Documents/ML_Assignment/Classification
python classification.py
```
*Outputs: Generates 12 figures in `Classification/figures/`, fits the Decision Tree (`max_depth=3`), saves `Classification/models/decision_tree_iris.joblib`, and tests predictions on unseen test samples.*

### 3. Execute Project B (Regression)
```bash
cd /home/yilu/Documents/ML_Assignment/Regression
python regression.py
```
*Outputs: Ingests `student-mat.csv`, generates 10 figures in `Regression/figures/`, compares 3 candidate configurations on validation data, trains the OLS Linear Regression model on Config 3, saves `Regression/models/linear_regression_student.joblib`, and tests predictions on student profiles.*

### 4. Launch Interactive Web Application Locally
```bash
cd /home/yilu/Documents/ML_Assignment
source Classification/.venv/bin/activate
streamlit run app.py
```
*Outputs: Starts local web server on `http://localhost:8501` allowing real-time interactive parameter adjustments and instant predictions for both Classification and Regression models.*

### 5. Deploying Online for Public Link Submission
To obtain a permanent, publicly shareable URL to submit to your instructor:
1. Initialize git and commit:
   ```bash
   git init
   git add .
   git commit -m "Machine Learning Assignment 1 Deployment"
   ```
2. Push to your GitHub account (e.g. `https://github.com/<your-username>/ml-assignment-1`).
3. Connect your repository to **[Streamlit Community Cloud](https://share.streamlit.io/)** (Free):
   - Click **New app** -> Select your repo -> Set main file to `app.py` -> Click **Deploy**.
   - Your live public submission link will be: `https://<your-username>-ml-assignment-1.streamlit.app`

---

## Executive Summary of Results

| Metric / Dimension | Project A: Classification (Iris) | Project B: Regression (Student Math) |
| :--- | :--- | :--- |
| **Dataset** | Fisher's Iris Dataset (150 samples) | UCI Student Performance (395 students) |
| **Target Variable** | Species (`setosa`, `versicolor`, `virginica`) | Final Mathematics Grade $G3$ ($0 - 20$) |
| **Algorithm** | Decision Tree Classifier (`gini`) | Ordinary Least Squares Linear Regression |
| **Data Partitioning** | 60% Train, 20% Val, 20% Test (Stratified) | 60% Train, 20% Val, 20% Test (Random) |
| **Hyperparameter Selection** | `max_depth = 3` (selected via validation F1) | Config 3: Expanded Socio-Academic Profile |
| **Primary Predictors** | Petal Length ($94.8\%$) & Petal Width ($5.2\%$) | Past Failures ($\beta = -2.01$), Higher Ed ($\beta = +0.80$), Medu ($\beta = +0.59$) |
| **Final Test Performance** | **100.00% Accuracy**, **1.0000 Macro F1** | **MAE = 3.39 points**, **RMSE = 4.43 points**, **$R^2 = 21.55\%$** |
| **Deployment Artifact** | `models/decision_tree_iris.joblib` ($2.68$ KB) | `models/linear_regression_student.joblib` ($1.44$ KB) |
| **Academic Report** | [Classification/report.md](file:///home/yilu/Documents/ML_Assignment/Classification/report.md) | [Regression/report.md](file:///home/yilu/Documents/ML_Assignment/Regression/report.md) |

