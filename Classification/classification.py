#!/usr/bin/env python3
"""
=============================================================================
Machine Learning Assignment 1 - Project A: Classification
Topic: Iris Flower Species Classification Using Decision Tree Classifier
Author: Yilkal Bewuketu (ID: GSR/2321/19)
Date: Academic Year 2026

Description:
This script performs a complete end-to-end Machine Learning pipeline on the
classic Iris dataset:
1. Data loading and exploratory data analysis (EDA)
2. Univariate, bivariate, and multivariate analysis with saved visual figures
3. Data preparation with a strict stratified 60/20/20 train/validation/test split
4. Decision Tree Classifier training and hyperparameter tuning (max_depth)
5. Rigorous model evaluation on unseen test data
6. Model persistence using joblib and sample prediction demonstration
=============================================================================
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Suppress minor non-critical pandas/seaborn formatting warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set global aesthetic styling for charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def setup_directories(base_dir: str):
    """Ensure output directories exist for figures and saved models."""
    figures_dir = os.path.join(base_dir, "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return figures_dir, models_dir


def load_and_understand_data():
    """
    Step 1: Load the Iris dataset and create a Pandas DataFrame.
    Inspect dimensions, data types, missing values, duplicates, and class balance.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA COLLECTION AND UNDERSTANDING")
    print("=" * 70)

    raw_iris = load_iris(as_frame=True)
    df = raw_iris.frame.copy()

    # Standardize column names for clean pythonic access
    column_mapping = {
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
        'target': 'target'
    }
    df = df.rename(columns=column_mapping)
    
    # Map integer target codes to human-readable biological species names
    target_names = [str(name) for name in raw_iris.target_names]  # ['setosa', 'versicolor', 'virginica']
    df['species'] = df['target'].map(lambda idx: target_names[idx])
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    print(f"Dataset Source: Fisher's Iris Dataset (via scikit-learn {load_iris.__module__})")
    print(f"Total Observations: {df.shape[0]} rows")
    print(f"Total Columns: {df.shape[1]} (4 continuous features, 1 integer target, 1 string label)")
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())

    print("\nDataset Info and Data Types:")
    df_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    print(df_info)

    print("\nDescriptive Statistics (Numerical Features):")
    stats = df[feature_names].describe().T
    stats['skewness'] = df[feature_names].skew()
    print(stats.round(3))

    # Duplicate check
    duplicate_count = df.duplicated(subset=feature_names).sum()
    print(f"\nDuplicate feature rows detected: {duplicate_count}")
    if duplicate_count > 0:
        print("Note: In the classic 150-sample Iris dataset, identical feature measurements exist")
        print("naturally for a few specimens. They represent valid biological observations.")

    # Class distribution
    print("\nTarget Class Distribution:")
    class_dist = df['species'].value_counts()
    for sp, cnt in class_dist.items():
        print(f"  - {sp}: {cnt} observations ({cnt / len(df):.1%})")

    return df, feature_names, target_names


def perform_univariate_analysis(df: pd.DataFrame, feature_names: list, figures_dir: str):
    """
    Step 2: Univariate Analysis.
    Analyze one feature at a time to examine distribution, central tendency, dispersion, and skew.
    """
    print("\n" + "=" * 70)
    print("STEP 2: UNIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating univariate visualizations: histograms, box plots, and class counts...")

    # 1. Histograms with KDE for all 4 numerical features
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, feature in enumerate(feature_names):
        ax = axes[i // 2, i % 2]
        sns.histplot(
            df[feature],
            kde=True,
            ax=ax,
            color=colors[i],
            bins=15,
            edgecolor='black',
            linewidth=0.8
        )
        mean_val = df[feature].mean()
        median_val = df[feature].median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='darkgreen', linestyle=':', label=f'Median: {median_val:.2f}')
        ax.set_title(f'Histogram & KDE: {feature.replace("_", " ").title()} (cm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Measurement (cm)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')

    plt.tight_layout()
    hist_path = os.path.join(figures_dir, "01_univariate_histograms.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Saved: {hist_path}")

    # 2. Box plots for all 4 numerical features (to highlight IQR and outliers)
    fig, ax = plt.subplots(figsize=(10, 6))
    melted_df = df.melt(value_vars=feature_names, var_name='Feature', value_name='Measurement (cm)')
    melted_df['Feature'] = melted_df['Feature'].apply(lambda x: x.replace("_", " ").title())
    sns.boxplot(
        x='Feature',
        y='Measurement (cm)',
        hue='Feature',
        data=melted_df,
        palette="Set2",
        legend=False,
        ax=ax,
        width=0.5,
        fliersize=5
    )
    ax.set_title('Univariate Box Plots: Distribution & Outlier Detection across Features', fontsize=13, fontweight='bold')
    ax.set_ylabel('Centimeters (cm)', fontsize=11)
    ax.set_xlabel('Botanical Feature', fontsize=11)
    plt.tight_layout()
    box_path = os.path.join(figures_dir, "02_univariate_boxplots.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"Saved: {box_path}")

    # 3. Species count plot
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {'setosa': '#4c72b0', 'versicolor': '#55a868', 'virginica': '#c44e52'}
    sns.countplot(
        x='species',
        hue='species',
        data=df,
        palette=palette,
        legend=False,
        ax=ax,
        edgecolor='black',
        linewidth=0.8
    )
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)} samples\n({int(height)/len(df):.1%})",
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.set_title('Univariate Target Distribution: Species Count', fontsize=13, fontweight='bold')
    ax.set_xlabel('Species', fontsize=11)
    ax.set_ylabel('Observation Count', fontsize=11)
    plt.tight_layout()
    count_path = os.path.join(figures_dir, "03_species_distribution.png")
    plt.savefig(count_path, dpi=300)
    plt.close()
    print(f"Saved: {count_path}")


def perform_bivariate_analysis(df: pd.DataFrame, feature_names: list, figures_dir: str):
    """
    Step 3: Bivariate Analysis.
    Analyze relationships between pairs of variables:
    - Sepal length vs sepal width
    - Petal length vs petal width
    - Each feature distributed across target classes
    - Pairwise correlations
    """
    print("\n" + "=" * 70)
    print("STEP 3: BIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating bivariate visualizations and calculating correlation statistics...")

    palette = {'setosa': '#4c72b0', 'versicolor': '#55a868', 'virginica': '#c44e52'}

    # 1. Sepal Length vs Sepal Width by species
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='sepal_length',
        y='sepal_width',
        hue='species',
        style='species',
        palette=palette,
        s=80,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title('Bivariate Scatter: Sepal Length vs. Sepal Width', fontsize=13, fontweight='bold')
    plt.xlabel('Sepal Length (cm)', fontsize=11)
    plt.ylabel('Sepal Width (cm)', fontsize=11)
    plt.legend(title='Species', loc='upper right')
    plt.tight_layout()
    sepal_scatter_path = os.path.join(figures_dir, "04_bivariate_sepal_length_vs_width.png")
    plt.savefig(sepal_scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {sepal_scatter_path}")

    # 2. Petal Length vs Petal Width by species
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x='petal_length',
        y='petal_width',
        hue='species',
        style='species',
        palette=palette,
        s=80,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.5
    )
    plt.title('Bivariate Scatter: Petal Length vs. Petal Width', fontsize=13, fontweight='bold')
    plt.xlabel('Petal Length (cm)', fontsize=11)
    plt.ylabel('Petal Width (cm)', fontsize=11)
    plt.legend(title='Species', loc='lower right')
    plt.tight_layout()
    petal_scatter_path = os.path.join(figures_dir, "05_bivariate_petal_length_vs_width.png")
    plt.savefig(petal_scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {petal_scatter_path}")

    # 3. Feature distributions compared across species (grouped boxplots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, feature in enumerate(feature_names):
        ax = axes[i // 2, i % 2]
        sns.boxplot(
            x='species',
            y=feature,
            data=df,
            hue='species',
            palette=palette,
            legend=False,
            ax=ax,
            width=0.5
        )
        ax.set_title(f'{feature.replace("_", " ").title()} by Species', fontsize=12, fontweight='bold')
        ax.set_xlabel('Species', fontsize=10)
        ax.set_ylabel('Centimeters (cm)', fontsize=10)
    plt.tight_layout()
    grouped_box_path = os.path.join(figures_dir, "06_features_by_species_boxplots.png")
    plt.savefig(grouped_box_path, dpi=300)
    plt.close()
    print(f"Saved: {grouped_box_path}")

    # 4. Pairwise correlation computation
    corr_matrix = df[feature_names].corr(method='pearson')
    print("\nPairwise Pearson Correlation Matrix:")
    print(corr_matrix.round(3))
    print("\nKey Bivariate Observation:")
    print("  - Petal length and Petal width have a very strong positive correlation (r = 0.963).")
    print("  - Setosa is linearly separable based solely on petal measurements.")
    print("  - Versicolor and Virginica show slight overlap in sepal dimensions but distinct petal dimensions.")


def perform_multivariate_analysis(df: pd.DataFrame, feature_names: list, figures_dir: str):
    """
    Step 4: Multivariate Analysis.
    Analyze joint interactions among three or more variables simultaneously:
    - Pair plot showing all feature pairs colored by species
    - Annotated Correlation Heatmap
    """
    print("\n" + "=" * 70)
    print("STEP 4: MULTIVARIATE ANALYSIS")
    print("=" * 70)
    print("Generating multivariate visualizations...")

    palette = {'setosa': '#4c72b0', 'versicolor': '#55a868', 'virginica': '#c44e52'}

    # 1. Pair Plot
    pairplot = sns.pairplot(
        df,
        vars=feature_names,
        hue='species',
        palette=palette,
        diag_kind='kde',
        plot_kws={'alpha': 0.8, 's': 40, 'edgecolor': 'none'},
        diag_kws={'fill': True}
    )
    pairplot.figure.suptitle('Multivariate Pair Plot: All Feature Interactions Across Species', y=1.02, fontsize=14, fontweight='bold')
    pairplot_path = os.path.join(figures_dir, "07_multivariate_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pairplot_path}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[feature_names].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Multivariate Feature Correlation Heatmap (Pearson)', fontsize=13, fontweight='bold')
    plt.xticks(ticks=np.arange(4) + 0.5, labels=[f.replace("_", " ").title() for f in feature_names], rotation=15)
    plt.yticks(ticks=np.arange(4) + 0.5, labels=[f.replace("_", " ").title() for f in feature_names], rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(figures_dir, "08_multivariate_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_path}")


def prepare_data(df: pd.DataFrame, feature_names: list):
    """
    Step 5: Data Preparation & Splitting.
    - Separate feature matrix X and target vector y
    - Partition dataset into stratified 60% Train, 20% Validation, 20% Test
    - Prevent data leakage: fitting operations must strictly use Train set
    """
    print("\n" + "=" * 70)
    print("STEP 5: DATA PREPARATION (REPRODUCIBLE 60/20/20 STRATIFIED SPLIT)")
    print("=" * 70)

    X = df[feature_names].copy()
    y = df['target'].copy()

    # Step A: Split 60% Train, 40% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    # Step B: Split the 40% Temp equally into 20% Validation and 20% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"Total Dataset Size:      {len(df)} samples")
    print(f"Training Set (60%):      {X_train.shape[0]} samples (30 per class)")
    print(f"Validation Set (20%):    {X_val.shape[0]} samples (10 per class)")
    print(f"Test Set (20%):          {X_test.shape[0]} samples (10 per class)")

    # Verify stratification
    print("\nClass Counts Verification across Splits:")
    val_check = pd.DataFrame({
        'Train (60%)': y_train.value_counts(),
        'Val (20%)': y_val.value_counts(),
        'Test (20%)': y_test.value_counts()
    })
    print(val_check)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         feature_names: list, target_names: list,
                         figures_dir: str):
    """
    Step 6: Algorithm Selection and Hyperparameter Tuning.
    - Model: Decision Tree Classifier
    - Compare candidate max_depth values: [1, 2, 3, 4, 5, 6, None]
    - Select optimal configuration using VALIDATION set metrics
    - Graph hyperparameter tuning performance curve
    - Plot feature importance and decision tree graph
    """
    print("\n" + "=" * 70)
    print("STEP 6: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("=" * 70)
    print("Algorithm: Decision Tree Classifier (Criterion: Gini Impurity)")
    print("Comparing max_depth candidates on Validation Set...")

    depth_candidates = [1, 2, 3, 4, 5, 6, None]
    results = []

    for depth in depth_candidates:
        dt = DecisionTreeClassifier(
            max_depth=depth,
            criterion='gini',
            random_state=42
        )
        dt.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, dt.predict(X_train))
        val_acc = accuracy_score(y_val, dt.predict(X_val))
        val_f1 = f1_score(y_val, dt.predict(X_val), average='macro')
        
        depth_label = "None (unlimited)" if depth is None else str(depth)
        results.append({
            'max_depth': depth_label,
            'depth_val': 99 if depth is None else depth,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1,
            'model': dt
        })

    tuning_df = pd.DataFrame(results)
    print("\nHyperparameter Search Results (Validation Performance):")
    display_df = tuning_df[['max_depth', 'train_accuracy', 'val_accuracy', 'val_f1_macro']].copy()
    print(display_df.to_string(index=False))

    # Graph validation curve for max_depth
    plt.figure(figsize=(9, 5))
    x_positions = range(len(depth_candidates))
    labels = [r['max_depth'] for r in results]
    plt.plot(x_positions, [r['train_accuracy'] for r in results], marker='o', linewidth=2, label='Train Accuracy', color='#1f77b4')
    plt.plot(x_positions, [r['val_accuracy'] for r in results], marker='s', linewidth=2, label='Validation Accuracy', color='#2ca02c')
    plt.plot(x_positions, [r['val_f1_macro'] for r in results], marker='^', linestyle='--', linewidth=2, label='Validation Macro F1', color='#ff7f0e')
    plt.xticks(x_positions, labels)
    plt.xlabel('Tree Maximum Depth (max_depth)', fontsize=11)
    plt.ylabel('Performance Metric Score', fontsize=11)
    plt.title('Hyperparameter Tuning: Decision Tree Depth vs. Model Accuracy', fontsize=13, fontweight='bold')
    plt.ylim(0.5, 1.05)
    plt.legend(loc='lower right')
    plt.tight_layout()
    tuning_plot_path = os.path.join(figures_dir, "09_hyperparameter_tuning_depth.png")
    plt.savefig(tuning_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {tuning_plot_path}")

    # Selection logic:
    # 1. Filter candidates that achieve maximum validation accuracy / F1
    # 2. Select the candidate with smallest max_depth to uphold Occam's Razor (prevent overfitting)
    max_val_acc = tuning_df['val_accuracy'].max()
    candidates = tuning_df[tuning_df['val_accuracy'] == max_val_acc]
    best_candidate = candidates.sort_values(by='depth_val').iloc[0]
    
    selected_depth = None if best_candidate['max_depth'] == "None (unlimited)" else int(best_candidate['max_depth'])
    print(f"\nOptimal Hyperparameter Selection based on Validation Set:")
    print(f"  Selected max_depth: {best_candidate['max_depth']}")
    print(f"  Validation Accuracy: {best_candidate['val_accuracy']:.4f}")
    print(f"  Validation Macro F1: {best_candidate['val_f1_macro']:.4f}")
    print(f"  Justification: Depth {best_candidate['max_depth']} achieves peak validation accuracy with minimum tree complexity.")

    # Final model using selected hyperparameter
    final_model = DecisionTreeClassifier(
        max_depth=selected_depth,
        criterion='gini',
        random_state=42
    )
    final_model.fit(X_train, y_train)

    # Feature Importance analysis
    importances = final_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': [f.replace("_", " ").title() for f in feature_names],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    print("\nDecision Tree Feature Importances:")
    for _, row in feat_imp_df.iloc[::-1].iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

    # Plot Feature Importance
    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#3470a3', edgecolor='black', height=0.55)
    plt.xlabel('Gini Feature Importance', fontsize=11)
    plt.title('Decision Tree Feature Importance (Gini Criterion)', fontsize=13, fontweight='bold')
    plt.xlim(0, max(importances) * 1.15)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f'{w:.3f} ({w*100:.1f}%)',
                 va='center', ha='left', fontsize=10, fontweight='bold')
    plt.tight_layout()
    imp_path = os.path.join(figures_dir, "10_feature_importance.png")
    plt.savefig(imp_path, dpi=300)
    plt.close()
    print(f"Saved: {imp_path}")

    # Plot Tree Structure
    plt.figure(figsize=(14, 8))
    plot_tree(
        final_model,
        feature_names=[f.replace("_", " ").title() for f in feature_names],
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f'Trained Decision Tree Architecture (max_depth={best_candidate["max_depth"]})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    tree_path = os.path.join(figures_dir, "11_decision_tree_structure.png")
    plt.savefig(tree_path, dpi=300)
    plt.close()
    print(f"Saved: {tree_path}")

    return final_model, selected_depth, tuning_df


def evaluate_final_model(model: DecisionTreeClassifier,
                         X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         target_names: list, figures_dir: str):
    """
    Step 7: Final Model Evaluation on Unseen Test Data.
    - Evaluate test accuracy, precision, recall, F1-scores
    - Display classification report and confusion matrix
    - Generate Confusion Matrix heatmap
    - Check for overfitting and underfitting across splits
    """
    print("\n" + "=" * 70)
    print("STEP 7: FINAL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Compute metrics for Test Set
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec_macro = precision_score(y_test, y_test_pred, average='macro')
    test_prec_weighted = precision_score(y_test, y_test_pred, average='weighted')
    test_rec_macro = recall_score(y_test, y_test_pred, average='macro')
    test_rec_weighted = recall_score(y_test, y_test_pred, average='weighted')
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    print("Final Model Test Set Metrics (Completely Unseen Data):")
    print(f"  - Accuracy:                 {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  - Precision (Macro avg):    {test_prec_macro:.4f}")
    print(f"  - Precision (Weighted avg): {test_prec_weighted:.4f}")
    print(f"  - Recall (Macro avg):       {test_rec_macro:.4f}")
    print(f"  - Recall (Weighted avg):    {test_rec_weighted:.4f}")
    print(f"  - F1-Score (Macro avg):     {test_f1_macro:.4f}")
    print(f"  - F1-Score (Weighted avg):  {test_f1_weighted:.4f}")

    print("\nDetailed Test Classification Report:")
    report = classification_report(y_test, y_test_pred, target_names=target_names, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix (Counts):")
    cm_df = pd.DataFrame(cm, index=[f"Actual {name}" for name in target_names],
                         columns=[f"Pred {name}" for name in target_names])
    print(cm_df)

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        annot_kws={'size': 14, 'fontweight': 'bold'}
    )
    plt.title('Test Set Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Species', fontsize=11)
    plt.ylabel('Actual Species', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "12_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # Generalization Assessment across Splits
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    print("\nGeneralization Analysis Across Dataset Partitions:")
    print(f"  - Training Accuracy (60%):   {train_acc:.4f}")
    print(f"  - Validation Accuracy (20%): {val_acc:.4f}")
    print(f"  - Test Accuracy (20%):       {test_acc:.4f}")
    
    diff = abs(train_acc - test_acc)
    if diff <= 0.05:
        print("  -> Conclusion: Excellent generalization. The model exhibits no substantial overfitting or underfitting.")
    elif train_acc > test_acc + 0.10:
        print("  -> Caution: Potential overfitting detected (training accuracy substantially exceeds test accuracy).")
    else:
        print("  -> Underfitting or variance present.")

    test_metrics = {
        'test_accuracy': test_acc,
        'precision_macro': test_prec_macro,
        'precision_weighted': test_prec_weighted,
        'recall_macro': test_rec_macro,
        'recall_weighted': test_rec_weighted,
        'f1_macro': test_f1_macro,
        'f1_weighted': test_f1_weighted,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'classification_report': report,
        'confusion_matrix': cm
    }
    return test_metrics


def save_model_artifact(model: DecisionTreeClassifier, feature_names: list,
                        target_names: list, selected_depth: int, models_dir: str):
    """
    Step 8: Model Serialization.
    Save the trained model and associated metadata bundle with joblib.
    """
    print("\n" + "=" * 70)
    print("STEP 8: MODEL SERIALIZATION (JOBLIB)")
    print("=" * 70)

    model_bundle = {
        'model': model,
        'model_type': 'DecisionTreeClassifier',
        'feature_names': feature_names,
        'target_names': target_names,
        'hyperparameters': {
            'max_depth': selected_depth,
            'criterion': 'gini',
            'random_state': 42
        },
        'metadata': {
            'problem': 'Iris Flower Classification',
            'library': 'scikit-learn',
            'created_by': 'Student & Instructor',
            'version': '1.0.0'
        }
    }

    model_filepath = os.path.join(models_dir, "decision_tree_iris.joblib")
    joblib.dump(model_bundle, model_filepath)
    print(f"Model artifact successfully saved to: {model_filepath}")
    file_size_kb = os.path.getsize(model_filepath) / 1024
    print(f"Artifact File Size: {file_size_kb:.2f} KB")

    return model_filepath


def predict_species(sepal_length: float, sepal_width: float,
                    petal_length: float, petal_width: float,
                    model_bundle_path: str) -> dict:
    """
    Standalone prediction utility function.
    Accepts 4 numerical flower measurements and returns predicted species + class probabilities.
    """
    bundle = joblib.load(model_bundle_path)
    model = bundle['model']
    feature_names = bundle['feature_names']
    target_names = bundle['target_names']

    input_df = pd.DataFrame([{
        'sepal_length': float(sepal_length),
        'sepal_width': float(sepal_width),
        'petal_length': float(petal_length),
        'petal_width': float(petal_width)
    }])[feature_names]

    pred_idx = model.predict(input_df)[0]
    pred_species = str(target_names[pred_idx])
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = {str(species): round(float(prob), 4) for species, prob in zip(target_names, probabilities)}

    return {
        'input_measurements': input_df.to_dict(orient='records')[0],
        'predicted_species': pred_species,
        'predicted_class_index': int(pred_idx),
        'probabilities': prob_dict
    }


def demonstrate_deployment_predictions(model_filepath: str, X_test: pd.DataFrame,
                                       y_test: pd.Series, target_names: list):
    """
    Step 9: Demonstration of Model Deployment & Prediction.
    Demonstrates model execution on unseen test samples representing each species.
    """
    print("\n" + "=" * 70)
    print("STEP 9: DEPLOYMENT DEMONSTRATION & SAMPLE INFERENCES")
    print("=" * 70)
    print("Loading serialized artifact and testing inference pipeline...")

    # Pick 1 sample of each species from test set for clear verification
    sample_indices = []
    for cls_idx in range(len(target_names)):
        idx = y_test[y_test == cls_idx].index[0]
        sample_indices.append(idx)

    print("\nRunning Predictions on Verified Unseen Botanical Specimens:")
    print("-" * 70)

    for i, idx in enumerate(sample_indices, 1):
        row = X_test.loc[idx]
        actual_label = target_names[y_test.loc[idx]]
        pred_result = predict_species(
            sepal_length=row['sepal_length'],
            sepal_width=row['sepal_width'],
            petal_length=row['petal_length'],
            petal_width=row['petal_width'],
            model_bundle_path=model_filepath
        )

        print(f"Sample {i} [Ground Truth: {actual_label}]:")
        print(f"  Measurements: Sepal L={row['sepal_length']}cm, Sepal W={row['sepal_width']}cm, "
              f"Petal L={row['petal_length']}cm, Petal W={row['petal_width']}cm")
        print(f"  Predicted Species:     >>> {pred_result['predicted_species'].upper()} <<<")
        print(f"  Class Probabilities:   {pred_result['probabilities']}")
        status = "MATCH (CORRECT)" if pred_result['predicted_species'] == actual_label else "MISMATCH"
        print(f"  Prediction Status:     {status}")
        print("-" * 70)


def main():
    """Main execution pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir, models_dir = setup_directories(base_dir)

    print("=" * 70)
    print("PROJECT A: IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING")
    print("=====================================================================")

    # 1. Load and understand data
    df, feature_names, target_names = load_and_understand_data()

    # 2. Univariate analysis
    perform_univariate_analysis(df, feature_names, figures_dir)

    # 3. Bivariate analysis
    perform_bivariate_analysis(df, feature_names, figures_dir)

    # 4. Multivariate analysis
    perform_multivariate_analysis(df, feature_names, figures_dir)

    # 5. Data preparation (60/20/20 stratified split)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(df, feature_names)

    # 6. Model training & hyperparameter tuning
    model, selected_depth, tuning_df = train_and_tune_model(
        X_train, y_train, X_val, y_val, feature_names, target_names, figures_dir
    )

    # 7. Model evaluation on test set
    test_metrics = evaluate_final_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test, target_names, figures_dir
    )

    # 8. Model serialization
    model_filepath = save_model_artifact(model, feature_names, target_names, selected_depth, models_dir)

    # 9. Deployment demonstration
    demonstrate_deployment_predictions(model_filepath, X_test, y_test, target_names)

    print("\n" + "=" * 70)
    print("PROJECT A EXECUTION COMPLETED SUCCESSFULLY!")
    print("=====================================================================")
    return {
        'df': df,
        'selected_depth': selected_depth,
        'test_metrics': test_metrics,
        'model_filepath': model_filepath
    }


if __name__ == '__main__':
    main()
