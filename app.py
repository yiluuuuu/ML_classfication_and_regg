#!/usr/bin/env python3
"""
=============================================================================
Machine Learning Assignment 1 - Interactive Web Application
Author: Yilkal Bewuketu (ID: GSR/2321/19)
Course: Machine Learning Assignment 1
Description:
A unified interactive web application deploying both:
1. Project A: Iris Flower Species Classification (Decision Tree Classifier)
2. Project B: Secondary Student Mathematics Grade Prediction (Linear Regression)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="ML Assignment 1 | Yilkal Bewuketu",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIF_MODEL_PATH = os.path.join(BASE_DIR, "Classification", "models", "decision_tree_iris.joblib")
REGRESS_MODEL_PATH = os.path.join(BASE_DIR, "Regression", "models", "linear_regression_student.joblib")


@st.cache_resource
def load_classification_model():
    if os.path.exists(CLASSIF_MODEL_PATH):
        return joblib.load(CLASSIF_MODEL_PATH)
    return None


@st.cache_resource
def load_regression_model():
    if os.path.exists(REGRESS_MODEL_PATH):
        return joblib.load(REGRESS_MODEL_PATH)
    return None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("ML Assignment 1")
    st.markdown("**Student:** Yilkal Bewuketu  \n**Student ID:** GSR/2321/19")
    st.markdown("---")
    
    selected_page = st.radio(
        "Choose Project Pipeline:",
        ["🌸 Project A: Iris Classification", "📊 Project B: Student Grade Regression", "ℹ️ Project & Model Info"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Summary")
    st.metric("Iris Test Accuracy", "100.0%", "Decision Tree (depth=3)")
    st.metric("Grade Prediction Test MAE", "3.39 pts", "OLS Linear Regression")
    st.metric("Grade Prediction R²", "21.55%", "6 Predictors (No Leakage)")
    
    st.markdown("---")
    st.caption("Submitted for Academic Evaluation | Academic Year 2026")


# =============================================================================
# PAGE 1: PROJECT A - IRIS FLOWER SPECIES CLASSIFICATION
# =============================================================================
if selected_page == "🌸 Project A: Iris Classification":
    st.title("🌸 Project A: Botanical Iris Flower Species Classification")
    st.markdown(
        """
        This application uses a trained **Decision Tree Classifier (`max_depth=3`, Gini Impurity)** 
        to classify an iris flower into one of three species (*Iris setosa*, *Iris versicolor*, or *Iris virginica*) 
        based on four continuous morphological measurements.
        """
    )
    st.markdown("---")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Enter Flower Morphological Features")
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.5, value=3.0, step=0.1)
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
        petal_width  = st.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=1.3, step=0.1)

        predict_button = st.button("🔮 Predict Iris Species", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. Model Prediction & Probabilities")
        bundle = load_classification_model()

        if bundle is None:
            st.error(f"Classification model artifact not found at: `{CLASSIF_MODEL_PATH}`. Please run `python Classification/classification.py` first.")
        else:
            model = bundle['model']
            target_names = bundle['target_names']

            input_features = pd.DataFrame([{
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }])

            # Make prediction
            pred_class_idx = model.predict(input_features)[0]
            pred_species = target_names[pred_class_idx]
            probabilities = model.predict_proba(input_features)[0]

            species_emojis = {
                'setosa': "🌱",
                'versicolor': "🌿",
                'virginica': "🌺"
            }

            st.success(f"### Predicted Species: {species_emojis.get(pred_species, '🌸')} **{pred_species.capitalize()}**")
            st.write(f"**Classification Confidence:** `{probabilities[pred_class_idx] * 100:.1f}%`")

            # Probability Distribution Chart
            prob_df = pd.DataFrame({
                "Species": [s.capitalize() for s in target_names],
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Species"))

            # Decision Logic Breakdown
            st.markdown("#### 🔍 Tree Decision Path Insight")
            if petal_length <= 2.45:
                st.info("💡 **Reasoning:** Petal length $\\le 2.45$ cm cleanly identifies **Setosa** with 100% purity.")
            elif petal_width <= 1.75:
                st.info("💡 **Reasoning:** Petal length $> 2.45$ cm and petal width $\\le 1.75$ cm leads predominantly to **Versicolor**.")
            else:
                st.info("💡 **Reasoning:** Petal width $> 1.75$ cm assigns the specimen into **Virginica**.")

    st.markdown("---")
    st.subheader("3. Model Architecture & Feature Importance")
    fig_col1, fig_col2 = st.columns(2)
    tree_fig_path = os.path.join(BASE_DIR, "Classification", "figures", "11_decision_tree_structure.png")
    imp_fig_path = os.path.join(BASE_DIR, "Classification", "figures", "10_feature_importance.png")

    with fig_col1:
        if os.path.exists(tree_fig_path):
            st.image(tree_fig_path, caption="Fitted Decision Tree Hierarchy (max_depth=3)")
    with fig_col2:
        if os.path.exists(imp_fig_path):
            st.image(imp_fig_path, caption="Gini Feature Importance (Petal length dominates with ~94.8%)")


# =============================================================================
# PAGE 2: PROJECT B - STUDENT EXAM SCORE REGRESSION
# =============================================================================
elif selected_page == "📊 Project B: Student Grade Regression":
    st.title("📊 Project B: Secondary Student Mathematics Grade Prediction")
    st.markdown(
        """
        This application deploys an **Ordinary Least Squares (OLS) Linear Regression** model trained on the 
        UCI Student Performance dataset. It predicts a student's final mathematics grade ($G3$, on a $0 - 20$ scale) 
        using socio-academic factors—**strictly excluding interim grades ($G1, G2$)** to serve as a genuine early-warning system.
        """
    )
    st.markdown("---")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Enter Student Profile Indicators")

        studytime_mapping = {
            "1 (< 2 hours / week)": 1,
            "2 (2 to 5 hours / week)": 2,
            "3 (5 to 10 hours / week)": 3,
            "4 (> 10 hours / week)": 4
        }
        studytime_label = st.selectbox("Weekly Study Time:", list(studytime_mapping.keys()), index=1)
        studytime_val = studytime_mapping[studytime_label]

        failures_val = st.slider("Past Class Failures (0 to 4):", min_value=0, max_value=4, value=0)
        absences_val = st.slider("Total School Absences:", min_value=0, max_value=50, value=4)

        edu_mapping = {
            "0: None": 0,
            "1: Primary (4th grade)": 1,
            "2: 5th to 9th grade": 2,
            "3: Secondary Education": 3,
            "4: Higher / University": 4
        }
        medu_label = st.selectbox("Mother's Education (Medu):", list(edu_mapping.keys()), index=3)
        medu_val = edu_mapping[medu_label]

        fedu_label = st.selectbox("Father's Education (Fedu):", list(edu_mapping.keys()), index=2)
        fedu_val = edu_mapping[fedu_label]

        higher_val = st.radio("Desires Higher Education?", ["Yes", "No"], horizontal=True, index=0)
        higher_encoded = 1 if higher_val == "Yes" else 0

        predict_grade_btn = st.button("📈 Predict Mathematics Grade (G3)", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. Predicted Final Mathematics Grade")
        reg_bundle = load_regression_model()

        if reg_bundle is None:
            st.error(f"Regression model artifact not found at: `{REGRESS_MODEL_PATH}`. Please run `python Regression/regression.py` first.")
        else:
            model = reg_bundle['model']
            features = reg_bundle['features']
            intercept = reg_bundle['intercept']
            coeffs = reg_bundle['coefficients']

            input_data = pd.DataFrame([{
                'studytime': studytime_val,
                'failures': failures_val,
                'absences': absences_val,
                'Medu': medu_val,
                'Fedu': fedu_val,
                'higher_encoded': higher_encoded
            }])

            # Make prediction and clamp to valid 0-20 scale
            raw_pred = float(model.predict(input_data)[0])
            pred_score = max(0.0, min(20.0, raw_pred))

            # Performance classification
            if pred_score >= 14.0:
                grade_status = "🌟 Very Good / Distinction"
                status_color = "success"
            elif pred_score >= 10.0:
                grade_status = "✅ Passing Grade"
                status_color = "info"
            else:
                grade_status = "⚠️ Academic Risk (Below Pass Threshold of 10/20)"
                status_color = "warning"

            st.metric(
                label="Predicted Final Exam Grade (G3 / 20 points)",
                value=f"{pred_score:.2f} / 20.0",
                delta=f"{pred_score - 10.0:+.2f} relative to pass line (10.0)"
            )

            if status_color == "success":
                st.success(f"**Academic Assessment:** {grade_status}")
            elif status_color == "info":
                st.info(f"**Academic Assessment:** {grade_status}")
            else:
                st.warning(f"**Academic Assessment:** {grade_status}")

            # Feature Contribution Breakdown
            st.markdown("#### 🧮 Model Equation Contribution Breakdown")
            contributions = {
                "Base (Intercept)": intercept,
                "Study Time (β=+0.43)": coeffs.get('studytime', 0.43) * studytime_val,
                "Failures (β=-2.01)": coeffs.get('failures', -2.01) * failures_val,
                "Absences (β=+0.06)": coeffs.get('absences', 0.06) * absences_val,
                "Mother Edu (β=+0.59)": coeffs.get('Medu', 0.59) * medu_val,
                "Father Edu (β=-0.21)": coeffs.get('Fedu', -0.21) * fedu_val,
                "Higher Ed Goal (β=+0.80)": coeffs.get('higher_encoded', 0.80) * higher_encoded,
            }
            contrib_df = pd.DataFrame(list(contributions.items()), columns=["Factor", "Impact (Points)"])
            st.dataframe(contrib_df.style.format({"Impact (Points)": "{:+.2f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Regression Insights & Diagnostic Plots")
    fig_col1, fig_col2 = st.columns(2)
    coeff_fig_path = os.path.join(BASE_DIR, "Regression", "figures", "10_regression_coefficients.png")
    pred_act_path = os.path.join(BASE_DIR, "Regression", "figures", "08_predicted_vs_actual_test.png")

    with fig_col1:
        if os.path.exists(coeff_fig_path):
            st.image(coeff_fig_path, caption="Learned Linear Regression Coefficients (Past failures have the largest negative penalty)")
    with fig_col2:
        if os.path.exists(pred_act_path):
            st.image(pred_act_path, caption="Predicted vs. Actual Exam Scores on Unseen Test Partition")


# =============================================================================
# PAGE 3: PROJECT & ACADEMIC DOCUMENTATION
# =============================================================================
else:
    st.title("ℹ️ Academic Project Documentation")
    st.markdown(
        """
        ### Machine Learning Assignment 1
        - **Student Name:** Yilkal Bewuketu
        - **Student ID:** GSR/2321/19
        - **Academic Year:** 2026
        - **Frameworks:** Python 3, scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn, Joblib

        ---
        ### 1. Project A: Multi-Class Iris Classification
        - **Algorithm:** Decision Tree Classifier (`criterion='gini'`, `max_depth=3`)
        - **Dataset:** Fisher's Iris Dataset (UCI Machine Learning Repository / scikit-learn)
        - **Data Split:** 60% Training, 20% Validation, 20% Testing (Stratified)
        - **Key Result:** 100% Test Accuracy, 1.0000 Macro F1.
        - **Dominant Feature:** Petal Length (94.8% Gini importance).

        ---
        ### 2. Project B: Student Exam Performance Regression
        - **Algorithm:** Ordinary Least Squares (OLS) Linear Regression
        - **Dataset:** UCI Student Performance Dataset (`student-mat.csv`, 395 samples)
        - **Data Split:** 60% Training, 20% Validation, 20% Testing (Random Seed 42)
        - **Key Result:** Test MAE = 3.39 points, RMSE = 4.43 points, $R^2 = 21.55\%$.
        - **Primary Driver:** Prior academic failures ($\beta = -2.01$ points per failed course).
        """
    )

