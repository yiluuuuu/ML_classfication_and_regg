# Project A: Iris Flower Species Classification Using Machine Learning

**Academic Coursework:** Machine Learning Assignment 1  
**Student Name:** Yilkal Bewuketu  
**Student ID:** GSR/2321/19  
**Problem Domain:** Supervised Multi-Class Classification  
**Target Variable:** Iris Species (`setosa`, `versicolor`, `virginica`)  
**Algorithm:** Decision Tree Classifier  
**Python Version:** Python 3.10+ (Tested on Python 3.14)

---

## 1. Project Overview

This project implements an end-to-end Machine Learning classification pipeline using the classic Iris botanical dataset. The objective is to identify the botanical species of an iris flower based on four morphological measurements:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

The project follows standard machine learning methodology:
- Exploratory Data Analysis (Univariate, Bivariate, and Multivariate)
- Reproducible Stratified 60/20/20 Train/Validation/Test Partitioning
- Hyperparameter Tuning for `max_depth` using strictly Validation Set metrics to prevent data leakage
- Rigorous final evaluation on unseen Test Data (Accuracy, Precision, Recall, Macro/Weighted F1, Confusion Matrix)
- Model serialization via `joblib` and automated inference demonstration

---

## 2. Directory Structure

```text
Classification/
├── .venv/                      # Virtual environment
├── figures/                    # 12 high-resolution analytical visual figures
│   ├── 01_univariate_histograms.png
│   ├── 02_univariate_boxplots.png
│   ├── 03_species_distribution.png
│   ├── 04_bivariate_sepal_length_vs_width.png
│   ├── 05_bivariate_petal_length_vs_width.png
│   ├── 06_features_by_species_boxplots.png
│   ├── 07_multivariate_pairplot.png
│   ├── 08_multivariate_correlation_heatmap.png
│   ├── 09_hyperparameter_tuning_depth.png
│   ├── 10_feature_importance.png
│   ├── 11_decision_tree_structure.png
│   └── 12_confusion_matrix.png
├── models/                     # Serialized model artifact
│   └── decision_tree_iris.joblib
├── classification.py           # Complete end-to-end Python pipeline
├── report.md                   # Full 24-section university academic report
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation and execution instructions
```

---

## 3. Installation and Setup

### Step A: Navigate to the Classification Directory
```bash
cd /home/yilu/Documents/ML_Assignment/Classification
```

### Step B: Create and Activate the Virtual Environment
If not already active:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step C: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Execution Instructions

To execute the entire training, evaluation, figure generation, and inference pipeline, run:

```bash
python classification.py
```

### What the script executes:
1. **Data Ingestion:** Loads Fisher's Iris dataset from `scikit-learn` (150 samples, 4 continuous features, 3 balanced classes of 50 samples each).
2. **Data Exploration:** Generates summary statistics, checks for missing values, and checks duplicate rows.
3. **Univariate Analysis:** Creates histograms with KDE curves, boxplots, and target distribution plots (`figures/01-03`).
4. **Bivariate Analysis:** Creates scatter plots and grouped boxplots, and calculates Pearson correlation (`figures/04-06`).
5. **Multivariate Analysis:** Generates pair plots and correlation heatmaps (`figures/07-08`).
6. **Data Partitioning:** Splits data into stratified 60% Train (90 samples), 20% Validation (30 samples), and 20% Test (30 samples).
7. **Hyperparameter Tuning:** Evaluates `max_depth` $\in \{1, 2, 3, 4, 5, 6, \text{None}\}$ on the validation set (`figures/09`), selecting optimal `max_depth = 3`.
8. **Feature Importance & Tree Graphing:** Analyzes Gini importance (`figures/10`) and plots the decision tree architecture (`figures/11`).
9. **Unseen Test Evaluation:** Evaluates the chosen model on the held-out test set and plots the confusion matrix (`figures/12`).
10. **Model Serialization:** Saves the model bundle to `models/decision_tree_iris.joblib`.
11. **Inference Demonstration:** Loads the serialized artifact and tests inference on sample botanical measurements.

---

## 5. Summary of Experimental Results

- **Selected Hyperparameter:** `max_depth = 3` (Criterion: Gini Impurity)
- **Validation Accuracy:** 96.67%
- **Validation Macro F1:** 0.9666
- **Test Set Accuracy (Unseen Data):** **100.00%** (30/30 correct)
- **Test Macro Precision:** 1.0000
- **Test Macro Recall:** 1.0000
- **Test Macro F1-Score:** 1.0000
- **Primary Discriminative Feature:** Petal Length (94.76% Gini importance)

---

## 6. How to Use the Saved Model for New Predictions

You can import and use the trained model bundle in any Python application:

```python
import joblib
import pandas as pd

# Load saved bundle
bundle = joblib.load("models/decision_tree_iris.joblib")
model = bundle['model']
target_names = bundle['target_names']

# Predict for a new flower specimen: [sepal_length, sepal_width, petal_length, petal_width]
new_sample = pd.DataFrame([{
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
}])

prediction_idx = model.predict(new_sample)[0]
print(f"Predicted Species: {target_names[prediction_idx]}")
```

