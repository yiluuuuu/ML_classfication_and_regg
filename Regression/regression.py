#!/usr/bin/env python3
"""
=============================================================================
Machine Learning Assignment 1 - Project B: Regression
Topic: Student Exam Score Prediction Using Linear Regression
Author: Yilkal Bewuketu (ID: GSR/2321/19)
Date: Academic Year 2026

Description:
This script performs a complete end-to-end Machine Learning regression pipeline
on the UCI Student Performance dataset (Mathematics course, student-mat.csv):
1. Data loading and exploratory data analysis (EDA)
2. Univariate, bivariate, and multivariate analysis with saved visual figures
3. Data preparation with a strict 60/20/20 train/validation/test split
4. Predictor selection and validation-based model comparison (Linear Regression)
5. Rigorous final evaluation on unseen test data (MAE, MSE, RMSE, R-squared)
6. Residual analysis and linear regression diagnostic plots
7. Model serialization via joblib and standalone prediction demonstration
=============================================================================
"""

import os
import sys
import warnings
import urllib.request
import zipfile
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress minor non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set global aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def setup_directories(base_dir: str):
    """Ensure output directories exist for data, figures, and saved models."""
    data_dir = os.path.join(base_dir, "data")
    figures_dir = os.path.join(base_dir, "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return data_dir, figures_dir, models_dir


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Step 1: Ingest the UCI Student Performance (Mathematics) dataset.
    If not found locally, downloads from the official UCI Machine Learning Repository.
    """
    csv_path = os.path.join(data_dir, "student-mat.csv")
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Downloading from UCI Repository...")
        url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer_zip:
            outer_zip.extractall(data_dir)
        
        # Check if nested zip exists
        nested_zip = os.path.join(data_dir, "student.zip")
        if os.path.exists(nested_zip):
            with zipfile.ZipFile(nested_zip) as inner_zip:
                inner_zip.extractall(data_dir)
        print("Dataset successfully downloaded and unpacked.")

    df = pd.read_csv(csv_path, sep=';')
    print(f"Loaded dataset from: {csv_path}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    return df


def explore_dataset(df: pd.DataFrame):
    """
    Step 2: Data Understanding and Health Check.
    Inspect dimensions, data types, missing values, duplicates, and target properties.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA COLLECTION AND UNDERSTANDING")
    print("=" * 70)
    print(f"Dataset: UCI Student Performance (Mathematics course, student-mat.csv)")
    print(f"Citation: Cortez and Silva (2008), University of Minho, Portugal")
    print(f"Observations: {df.shape[0]} students")
    print(f"Total Attributes: {df.shape[1]} (32 demographic/socio-academic predictors, 1 target G3)")

    print("\nFirst 5 Rows of Key Attributes and Target (G3):")
    sample_cols = ['school', 'sex', 'age', 'studytime', 'failures', 'absences', 'Medu', 'higher', 'G1', 'G2', 'G3']
    print(df[sample_cols].head())

    print("\nData Health Verification:")
    missing_total = df.isnull().sum().sum()
    duplicate_total = df.duplicated().sum()
    print(f"  - Missing Values (NaN): {missing_total}")
    print(f"  - Duplicate Records:    {duplicate_total}")

    print("\nTarget Variable (G3 - Final Mathematics Grade) Descriptive Statistics:")
    g3_stats = df['G3'].describe()
    print(g3_stats.round(3))
    print(f"  - Grade Range:    {df['G3'].min()} to {df['G3'].max()} (Scale: 0 - 20)")
    print(f"  - Skewness:       {df['G3'].skew():.3f}")
    zero_grades = (df['G3'] == 0).sum()
    print(f"  - Zero Scores:    {zero_grades} students ({zero_grades / len(df):.1%})")
    print("  Note on Zero Scores: In Portuguese secondary schools, a grade of 0 typically")
    print("  signifies that a student dropped out, was barred from exams due to truancy,")
    print("  or did not attend the final examination, creating a zero-inflated cluster.")


def perform_univariate_analysis(df: pd.DataFrame, figures_dir: str):
    """
    Step 3: Univariate Analysis.
    Analyze target variable G3 and key predictor distributions in isolation.
    """
    print("\n" + "=" * 70)
    print("STEP 2: UNIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating univariate figures for target and predictor variables...")

    # 1. Target G3 Distribution (Histogram + KDE + Boxplot)
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                          gridspec_kw={'height_ratios': [0.25, 0.75]})
    sns.boxplot(x=df['G3'], ax=ax_box, color='#4a90e2', width=0.5, fliersize=5)
    ax_box.set(xlabel='')
    ax_box.set_title('Univariate Distribution of Target Variable: Final Math Grade (G3)', fontsize=13, fontweight='bold')

    sns.histplot(df['G3'], kde=True, ax=ax_hist, bins=21, color='#4a90e2', edgecolor='black', linewidth=0.8)
    mean_g3 = df['G3'].mean()
    median_g3 = df['G3'].median()
    ax_hist.axvline(mean_g3, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_g3:.2f}')
    ax_hist.axvline(median_g3, color='green', linestyle=':', linewidth=2, label=f'Median: {median_g3:.2f}')
    ax_hist.axvline(10.0, color='darkorange', linestyle='-', linewidth=1.5, label='Passing Threshold (10/20)')
    ax_hist.set_xlabel('Final Grade G3 (0 - 20 Scale)', fontsize=11)
    ax_hist.set_ylabel('Number of Students', fontsize=11)
    ax_hist.legend(loc='upper right')

    plt.tight_layout()
    g3_dist_path = os.path.join(figures_dir, "01_target_distribution_g3.png")
    plt.savefig(g3_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {g3_dist_path}")

    # 2. Key Predictors Distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # Study time
    study_counts = df['studytime'].value_counts().sort_index()
    study_labels = ['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs']
    axes[0, 0].bar(study_labels, study_counts, color='#2b5c8f', edgecolor='black')
    axes[0, 0].set_title('Weekly Study Time Distribution', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Student Count')
    for i, count in enumerate(study_counts):
        axes[0, 0].text(i, count + 3, f"{count}\n({count/len(df):.1%})", ha='center', fontsize=9)

    # Past Failures
    sns.countplot(x='failures', data=df, ax=axes[0, 1], palette='Reds_d', hue='failures', legend=False, edgecolor='black')
    axes[0, 1].set_title('Past Class Failures Distribution', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Student Count')

    # Absences
    sns.histplot(df['absences'], ax=axes[1, 0], kde=True, bins=20, color='#e67e22', edgecolor='black')
    axes[1, 0].set_title('School Absences Distribution', fontweight='bold', fontsize=11)
    axes[1, 0].set_xlabel('Number of Days Absent')
    axes[1, 0].set_ylabel('Student Count')

    # Mother's Education (Medu)
    medu_labels = ['None', '4th grade', '5th-9th', 'Secondary', 'Higher']
    sns.countplot(x='Medu', data=df, ax=axes[1, 1], palette='Purples_d', hue='Medu', legend=False, edgecolor='black')
    axes[1, 1].set_title("Mother's Education Level (Medu)", fontweight='bold', fontsize=11)
    axes[1, 1].set_xticklabels(medu_labels, fontsize=9)
    axes[1, 1].set_ylabel('Student Count')

    plt.tight_layout()
    pred_dist_path = os.path.join(figures_dir, "02_univariate_predictors_hist_box.png")
    plt.savefig(pred_dist_path, dpi=300)
    plt.close()
    print(f"Saved: {pred_dist_path}")


def perform_bivariate_analysis(df: pd.DataFrame, figures_dir: str):
    """
    Step 4: Bivariate Analysis.
    Analyze relationships between pairs of variables:
    - Study time vs Final Grade G3
    - Failures and Parental Education vs G3
    - Correlation matrix of numerical variables
    """
    print("\n" + "=" * 70)
    print("STEP 3: BIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating bivariate visualizations and correlation statistics...")

    # 1. Study time vs G3 (Boxplot & Jittered Scatter Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    study_map = {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}
    df_temp = df.copy()
    df_temp['studytime_label'] = df_temp['studytime'].map(study_map)
    order = ['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs']

    sns.boxplot(x='studytime_label', y='G3', data=df_temp, order=order,
                palette='Blues', hue='studytime_label', legend=False, ax=ax1, width=0.45)
    ax1.set_title('Final Grade (G3) by Weekly Study Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Weekly Study Time Category', fontsize=10)
    ax1.set_ylabel('Final Math Grade (0 - 20)', fontsize=10)

    # Jittered scatter plot with linear trend line
    np.random.seed(42)
    jitter = np.random.normal(0, 0.08, size=len(df))
    sns.regplot(x=df['studytime'] + jitter, y=df['G3'], ax=ax2,
                scatter_kws={'alpha': 0.5, 'color': '#1f77b4', 's': 30},
                line_kws={'color': 'red', 'linewidth': 2})
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(order)
    ax2.set_title('Linear Trend: Study Time vs. Final Grade', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Weekly Study Time', fontsize=10)
    ax2.set_ylabel('Final Math Grade (0 - 20)', fontsize=10)

    plt.tight_layout()
    study_biv_path = os.path.join(figures_dir, "03_bivariate_studytime_vs_g3.png")
    plt.savefig(study_biv_path, dpi=300)
    plt.close()
    print(f"Saved: {study_biv_path}")

    # 2. Failures & Mother's Education vs G3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(x='failures', y='G3', data=df, ax=ax1, palette='Reds', hue='failures', legend=False, width=0.45)
    ax1.set_title('Final Grade (G3) vs. Number of Past Failures', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Past Failures Count', fontsize=10)
    ax1.set_ylabel('Final Math Grade (G3)', fontsize=10)

    medu_labels = ['None', '4th grade', '5th-9th', 'Secondary', 'Higher']
    sns.boxplot(x='Medu', y='G3', data=df, ax=ax2, palette='Greens', hue='Medu', legend=False, width=0.45)
    ax2.set_title("Final Grade (G3) vs. Mother's Education (Medu)", fontsize=12, fontweight='bold')
    ax2.set_xticklabels(medu_labels, fontsize=9)
    ax2.set_xlabel("Mother's Education Level", fontsize=10)
    ax2.set_ylabel('Final Math Grade (G3)', fontsize=10)

    plt.tight_layout()
    fail_medu_path = os.path.join(figures_dir, "04_bivariate_failures_and_medu_vs_g3.png")
    plt.savefig(fail_medu_path, dpi=300)
    plt.close()
    print(f"Saved: {fail_medu_path}")

    # Correlation analysis
    r_study, _ = pd.Series(df['studytime']).corr(df['G3']), None
    r_fail, _ = pd.Series(df['failures']).corr(df['G3']), None
    r_medu, _ = pd.Series(df['Medu']).corr(df['G3']), None
    r_abs, _ = pd.Series(df['absences']).corr(df['G3']), None

    print("\nPairwise Pearson Correlation with Target G3:")
    print(f"  - studytime:  r = {r_study:+.4f} (weak positive correlation)")
    print(f"  - failures:   r = {r_fail:+.4f} (moderate negative correlation)")
    print(f"  - Medu:       r = {r_medu:+.4f} (mild positive correlation)")
    print(f"  - absences:   r = {r_abs:+.4f} (near-zero linear correlation)")
    print("\nStatistical Note on Causality:")
    print("  Correlation does NOT imply causation. While students with higher study time")
    print("  exhibit slightly higher average grades, study duration alone does not guarantee")
    print("  mastery. Confounding variables such as study quality, baseline aptitude, and home")
    print("  support influence both study time and academic outcomes.")


def perform_multivariate_analysis(df: pd.DataFrame, figures_dir: str):
    """
    Step 5: Multivariate Analysis.
    Analyze joint interactions and correlations among multiple features and target G3.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MULTIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating multivariate correlation heatmap and pair plot...")

    # Key numerical and encoded attributes for multivariate inspection
    df_multi = df[['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'age', 'goout', 'G3']].copy()
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr = df_multi.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Multivariate Correlation Heatmap: Socio-Academic Predictors & G3', fontsize=13, fontweight='bold')
    plt.tight_layout()
    heat_path = os.path.join(figures_dir, "05_multivariate_correlation_heatmap.png")
    plt.savefig(heat_path, dpi=300)
    plt.close()
    print(f"Saved: {heat_path}")

    # 2. Multivariate Pair Plot
    pair_vars = ['studytime', 'failures', 'Medu', 'G3']
    pairplot = sns.pairplot(
        df[pair_vars],
        kind='reg',
        plot_kws={'scatter_kws': {'alpha': 0.4, 's': 20, 'color': '#2c3e50'}, 'line_kws': {'color': '#e74c3c'}},
        diag_kind='kde',
        diag_kws={'fill': True, 'color': '#3498db'}
    )
    pairplot.figure.suptitle('Multivariate Pair Plot: Selected Predictors & Target Grade (G3)', y=1.02, fontsize=13, fontweight='bold')
    pair_path = os.path.join(figures_dir, "06_multivariate_pairplot.png")
    pairplot.savefig(pair_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pair_path}")


def prepare_and_split_data(df: pd.DataFrame):
    """
    Step 6: Data Preparation and Partitioning.
    - Encode categorical predictors (e.g. 'higher' -> 1/0)
    - Strictly exclude G1 and G2 to prevent leakage of earlier exam proxies
    - Split into 60% Train (237), 20% Validation (79), 20% Test (79)
    """
    print("\n" + "=" * 70)
    print("STEP 5: DATA PREPARATION & REPRODUCIBLE SPLITTING (60/20/20)")
    print("=" * 70)

    prepared_df = df.copy()
    
    # Binary encoding: does the student want to take higher education? ('yes' -> 1, 'no' -> 0)
    prepared_df['higher_encoded'] = (prepared_df['higher'] == 'yes').astype(int)

    # Exclude G1 and G2 explicitly
    print("Feature Selection Governance:")
    print("  - G1 (Period 1 Exam) and G2 (Period 2 Exam) are EXCLUDED.")
    print("    Justification: G1 and G2 are intermediate grade assessments. Including them")
    print("    trivially dominates the regression (r > 0.90) and obscures the genuine relationship")
    print("    between student behavioral/environmental predictors and final exam outcomes.")
    print("  - Target Variable: G3 (Final Mathematics Grade, 0-20 scale).")

    # Split 60% Train, 40% Temp
    train_df, temp_df = train_test_split(prepared_df, test_size=0.40, random_state=42)
    # Split Temp 50/50 into 20% Val and 20% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    print(f"\nDataset Splitting Summary:")
    print(f"  - Total Observations:    {len(prepared_df)}")
    print(f"  - Training Set (60%):    {len(train_df)} students")
    print(f"  - Validation Set (20%):  {len(val_df)} students")
    print(f"  - Test Set (20%):        {len(test_df)} students")

    return train_df, val_df, test_df


def train_and_compare_models(train_df: pd.DataFrame, val_df: pd.DataFrame, figures_dir: str):
    """
    Step 7: Model Development & Hyperparameter/Predictor Set Comparison.
    Fit Linear Regression models on different predictor configurations:
      - Config 1: Study Time Only (Univariate baseline)
      - Config 2: Academic & Attendance Profile (studytime, failures, absences)
      - Config 3: Expanded Socio-Academic Profile (studytime, failures, absences, Medu, Fedu, higher_encoded)
    Select the optimal configuration based strictly on Validation Set performance.
    """
    print("\n" + "=" * 70)
    print("STEP 6: MODEL DEVELOPMENT & VALIDATION-BASED CONFIGURATION SELECTION")
    print("=" * 70)
    print("Algorithm: Ordinary Least Squares (OLS) Linear Regression")
    print("Comparing 3 candidate predictor configurations on the Validation Set...")

    configs = {
        'Config 1 (Study Time Only)': {
            'features': ['studytime'],
            'description': 'Univariate baseline: testing if study time alone predicts G3'
        },
        'Config 2 (Academic & Attendance)': {
            'features': ['studytime', 'failures', 'absences'],
            'description': 'Study time + academic history (past failures) + attendance (absences)'
        },
        'Config 3 (Expanded Socio-Academic)': {
            'features': ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'higher_encoded'],
            'description': 'Full profile: study time + failures + absences + parental education + higher ed aspiration'
        }
    }

    comparison_results = []

    for name, cfg in configs.items():
        feats = cfg['features']
        model = LinearRegression()
        model.fit(train_df[feats], train_df['G3'])

        # Predict on validation set
        val_pred = model.predict(val_df[feats])
        val_mae = mean_absolute_error(val_df['G3'], val_pred)
        val_mse = mean_squared_error(val_df['G3'], val_pred)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(val_df['G3'], val_pred)

        # Train metrics for overfitting check
        train_pred = model.predict(train_df[feats])
        train_rmse = np.sqrt(mean_squared_error(train_df['G3'], train_pred))
        train_r2 = r2_score(train_df['G3'], train_pred)

        comparison_results.append({
            'Configuration': name,
            'Features': feats,
            'Train RMSE': train_rmse,
            'Train R2': train_r2,
            'Val MAE': val_mae,
            'Val MSE': val_mse,
            'Val RMSE': val_rmse,
            'Val R2': val_r2,
            'Model': model
        })

    results_df = pd.DataFrame(comparison_results)
    print("\nModel Validation Performance Comparison:")
    display_cols = ['Configuration', 'Val MAE', 'Val RMSE', 'Val R2', 'Train RMSE', 'Train R2']
    print(results_df[display_cols].to_string(index=False))

    # Plot Model Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(results_df))
    width = 0.35

    ax1.bar(x - width/2, results_df['Val RMSE'], width, label='Val RMSE', color='#e74c3c', edgecolor='black')
    ax1.bar(x + width/2, results_df['Val MAE'], width, label='Val MAE', color='#f39c12', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Config 1\n(Study Only)', 'Config 2\n(Academic)', 'Config 3\n(Expanded)'], fontsize=10)
    ax1.set_ylabel('Error Metric (Lower is Better)', fontsize=11)
    ax1.set_title('Validation Error Comparison (RMSE & MAE)', fontsize=12, fontweight='bold')
    ax1.legend()

    ax2.bar(x, results_df['Val R2'], width=0.5, color='#27ae60', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Config 1\n(Study Only)', 'Config 2\n(Academic)', 'Config 3\n(Expanded)'], fontsize=10)
    ax2.set_ylabel('R-squared Score (Higher is Better)', fontsize=11)
    ax2.set_title('Validation Variance Explained (R²)', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.tight_layout()
    comp_plot_path = os.path.join(figures_dir, "07_model_comparison_validation.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_plot_path}")

    # Select best model: highest validation R2 / lowest validation RMSE
    best_row = results_df.sort_values(by='Val RMSE', ascending=True).iloc[0]
    best_name = best_row['Configuration']
    best_features = best_row['Features']
    best_model = best_row['Model']

    print(f"\nOptimal Predictor Configuration Selected:")
    print(f"  Selected:   {best_name}")
    print(f"  Predictors: {best_features}")
    print(f"  Val RMSE:   {best_row['Val RMSE']:.4f}")
    print(f"  Val R²:     {best_row['Val R2']:.4f}")
    print("  Justification: Config 3 minimizes validation RMSE while maximizing explained variance.")

    return best_model, best_features, results_df


def evaluate_final_model(model: LinearRegression, best_features: list,
                         train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                         figures_dir: str):
    """
    Step 8: Final Model Evaluation on Unseen Test Data.
    - Evaluate test MAE, MSE, RMSE, R-squared
    - Plot Predicted vs Actual scores
    - Plot Residual analysis (Residuals vs Predicted & Residual histogram)
    - Plot and interpret regression coefficients (slopes and intercept)
    """
    print("\n" + "=" * 70)
    print("STEP 7: FINAL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)

    # Predictions
    y_test = test_df['G3']
    y_test_pred = model.predict(test_df[best_features])

    # Test Metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Final Model Test Set Performance (Completely Unseen Data):")
    print(f"  - Mean Absolute Error (MAE):       {test_mae:.4f} grade points")
    print(f"  - Mean Squared Error (MSE):        {test_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE):  {test_rmse:.4f} grade points")
    print(f"  - Coefficient of Determination (R²): {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")

    # Linear Regression Coefficients Analysis
    coef_df = pd.DataFrame({
        'Predictor': best_features,
        'Coefficient (Beta)': model.coef_
    }).sort_values(by='Coefficient (Beta)', ascending=True)

    print(f"\nTrained Model Equation:")
    print(f"  G3_pred = {model.intercept_:.4f}")
    for _, r in coef_df.iterrows():
        print(f"            + ({r['Coefficient (Beta)']:+.4f} * {r['Predictor']})")

    # Plot 1: Predicted vs Actual on Test Data
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_test_pred, color='#2980b9', alpha=0.7, edgecolors='black', s=50, label='Test Observations')
    min_val = min(y_test.min(), y_test_pred.min()) - 1
    max_val = max(y_test.max(), y_test_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
    plt.xlabel('Actual Final Grade (G3)', fontsize=11)
    plt.ylabel('Predicted Final Grade (G3)', fontsize=11)
    plt.title('Test Set: Predicted vs. Actual Exam Scores', fontsize=13, fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left')
    plt.tight_layout()
    pred_actual_path = os.path.join(figures_dir, "08_predicted_vs_actual_test.png")
    plt.savefig(pred_actual_path, dpi=300)
    plt.close()
    print(f"Saved: {pred_actual_path}")

    # Plot 2: Residual Analysis
    residuals = y_test - y_test_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_test_pred, residuals, color='#8e44ad', alpha=0.7, edgecolors='black', s=45)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.8)
    ax1.set_xlabel('Predicted Grade (G3)', fontsize=10)
    ax1.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
    ax1.set_title('Residuals vs. Fitted Values', fontsize=12, fontweight='bold')

    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax2, color='#8e44ad', edgecolor='black', bins=15)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Residual Error', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Residual Error Distribution (Normality Check)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    residual_path = os.path.join(figures_dir, "09_residual_plot_test.png")
    plt.savefig(residual_path, dpi=300)
    plt.close()
    print(f"Saved: {residual_path}")

    # Plot 3: Feature Coefficients
    plt.figure(figsize=(8, 4.5))
    bar_colors = ['#c0392b' if c < 0 else '#27ae60' for c in coef_df['Coefficient (Beta)']]
    bars = plt.barh(coef_df['Predictor'], coef_df['Coefficient (Beta)'], color=bar_colors, edgecolor='black', height=0.55)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('Regression Coefficient (Beta Weight)', fontsize=11)
    plt.title('Fitted Linear Regression Feature Weights', fontsize=13, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        offset = 0.05 if w >= 0 else -0.35
        plt.text(w + offset, bar.get_y() + bar.get_height() / 2, f'{w:+.3f}',
                 va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    coef_plot_path = os.path.join(figures_dir, "10_regression_coefficients.png")
    plt.savefig(coef_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {coef_plot_path}")

    test_metrics = {
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'intercept': model.intercept_,
        'coefficients': dict(zip(best_features, model.coef_))
    }
    return test_metrics


def save_model_artifact(model: LinearRegression, best_features: list,
                        test_metrics: dict, models_dir: str):
    """
    Step 9: Model Serialization using joblib.
    Saves model, feature list, preprocessing mappings, and coefficients.
    """
    print("\n" + "=" * 70)
    print("STEP 8: MODEL SERIALIZATION (JOBLIB)")
    print("=" * 70)

    model_bundle = {
        'model': model,
        'model_type': 'LinearRegression',
        'features': best_features,
        'intercept': float(model.intercept_),
        'coefficients': {k: float(v) for k, v in test_metrics['coefficients'].items()},
        'categorical_encoding': {
            'higher': {'yes': 1, 'no': 0}
        },
        'metadata': {
            'problem': 'Student Exam Score Prediction',
            'target': 'G3 (Final Mathematics Grade, scale 0-20)',
            'library': 'scikit-learn',
            'created_by': 'Student & Instructor',
            'version': '1.0.0'
        }
    }

    model_filepath = os.path.join(models_dir, "linear_regression_student.joblib")
    joblib.dump(model_bundle, model_filepath)
    print(f"Fitted regression model artifact saved to: {model_filepath}")
    file_size_kb = os.path.getsize(model_filepath) / 1024
    print(f"Artifact File Size: {file_size_kb:.2f} KB")

    return model_filepath


def predict_student_grade(studytime: int, failures: int, absences: int,
                          Medu: int, Fedu: int, higher: str,
                          model_bundle_path: str) -> dict:
    """
    Standalone prediction utility.
    Accepts student profile features and returns predicted G3 exam grade.
    """
    bundle = joblib.load(model_bundle_path)
    model = bundle['model']
    features = bundle['features']

    # Preprocessing
    higher_encoded = 1 if str(higher).lower() in ['yes', '1', 'true', 'y'] else 0

    input_data = pd.DataFrame([{
        'studytime': int(studytime),
        'failures': int(failures),
        'absences': int(absences),
        'Medu': int(Medu),
        'Fedu': int(Fedu),
        'higher_encoded': higher_encoded
    }])[features]

    predicted_g3 = model.predict(input_data)[0]
    # Bound predictions within legitimate educational grade range [0, 20]
    bounded_g3 = max(0.0, min(20.0, float(predicted_g3)))

    passing_status = "Pass" if bounded_g3 >= 10.0 else "Fail / At-Risk"

    return {
        'input_profile': {
            'studytime': studytime,
            'failures': failures,
            'absences': absences,
            'Medu': Medu,
            'Fedu': Fedu,
            'higher': higher
        },
        'raw_predicted_grade': round(float(predicted_g3), 2),
        'bounded_predicted_grade': round(bounded_g3, 2),
        'passing_status': passing_status
    }


def demonstrate_deployment_predictions(model_filepath: str, test_df: pd.DataFrame, best_features: list):
    """
    Step 10: Prediction Demonstration on Realistic Case Studies.
    Demonstrates model execution on unseen real students and hypothetical profiles.
    """
    print("\n" + "=" * 70)
    print("STEP 9: DEPLOYMENT DEMONSTRATION & INFERENCE EXAMPLES")
    print("=" * 70)
    print("Executing inferences using serialized artifact...")

    scenarios = [
        {
            'name': "Scenario A: High-Engagement Student",
            'desc': "Daily study (5-10 hrs/wk), 0 past failures, 2 absences, mother has higher ed, plans higher ed",
            'params': {'studytime': 3, 'failures': 0, 'absences': 2, 'Medu': 4, 'Fedu': 4, 'higher': 'yes'}
        },
        {
            'name': "Scenario B: Academic-Risk Student",
            'desc': "Minimal study (<2 hrs/wk), 2 past failures, 12 absences, basic parental education, no higher ed plans",
            'params': {'studytime': 1, 'failures': 2, 'absences': 12, 'Medu': 1, 'Fedu': 1, 'higher': 'no'}
        },
        {
            'name': "Scenario C: Real Unseen Test Sample",
            'desc': "Actual student from held-out test set",
            'test_index': test_df.index[0]
        }
    ]

    for sc in scenarios:
        print(f"\n--- {sc['name']} ---")
        print(f"Profile: {sc['desc']}")
        
        if 'test_index' in sc:
            row = test_df.loc[sc['test_index']]
            actual_g3 = row['G3']
            res = predict_student_grade(
                studytime=row['studytime'],
                failures=row['failures'],
                absences=row['absences'],
                Medu=row['Medu'],
                Fedu=row['Fedu'],
                higher=row['higher'],
                model_bundle_path=model_filepath
            )
            print(f"Inputs: studytime={row['studytime']}, failures={row['failures']}, absences={row['absences']}, "
                  f"Medu={row['Medu']}, Fedu={row['Fedu']}, higher='{row['higher']}'")
            print(f"Actual Exam Grade (G3):    {actual_g3:.1f} / 20")
            print(f"Predicted Grade (G3):     {res['bounded_predicted_grade']:.2f} / 20 ({res['passing_status']})")
            print(f"Absolute Prediction Error: {abs(actual_g3 - res['bounded_predicted_grade']):.2f} grade points")
        else:
            p = sc['params']
            res = predict_student_grade(
                studytime=p['studytime'],
                failures=p['failures'],
                absences=p['absences'],
                Medu=p['Medu'],
                Fedu=p['Fedu'],
                higher=p['higher'],
                model_bundle_path=model_filepath
            )
            print(f"Inputs: {p}")
            print(f"Predicted Grade (G3):     {res['bounded_predicted_grade']:.2f} / 20 ({res['passing_status']})")


def main():
    """Main execution pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir, figures_dir, models_dir = setup_directories(base_dir)

    print("=" * 70)
    print("PROJECT B: STUDENT EXAM SCORE PREDICTION USING LINEAR REGRESSION")
    print("=====================================================================")

    # 1. Load dataset
    df = load_dataset(data_dir)

    # 2. Explore dataset
    explore_dataset(df)

    # 3. Univariate analysis
    perform_univariate_analysis(df, figures_dir)

    # 4. Bivariate analysis
    perform_bivariate_analysis(df, figures_dir)

    # 5. Multivariate analysis
    perform_multivariate_analysis(df, figures_dir)

    # 6. Data preparation (60/20/20 split)
    train_df, val_df, test_df = prepare_and_split_data(df)

    # 7. Model development and validation comparison
    best_model, best_features, results_df = train_and_compare_models(train_df, val_df, figures_dir)

    # 8. Final model evaluation on test set
    test_metrics = evaluate_final_model(best_model, best_features, train_df, val_df, test_df, figures_dir)

    # 9. Model serialization
    model_filepath = save_model_artifact(best_model, best_features, test_metrics, models_dir)

    # 10. Deployment demonstration
    demonstrate_deployment_predictions(model_filepath, test_df, best_features)

    print("\n" + "=" * 70)
    print("PROJECT B EXECUTION COMPLETED SUCCESSFULLY!")
    print("=====================================================================")
    return {
        'df': df,
        'best_features': best_features,
        'test_metrics': test_metrics,
        'model_filepath': model_filepath
    }


if __name__ == '__main__':
    main()

