# Machine Learning Assignment 1: Project B Report
# Student Exam Score Prediction Using Linear Regression

**Course:** Machine Learning Assignment 1  
**Student Name:** Yilkal Bewuketu  
**Student ID:** GSR/2321/19  
**Supervisor / Instructor:** Course Instructor  
**Date:** Academic Year 2026  
**Artifact Directory:** `Regression/`  
**Execution Environment:** Python 3.14 / scikit-learn 1.9.1 / Pandas 3.1.1 / Matplotlib 3.11.2 / Seaborn 0.13.2  

---

## 1. Title
**Predicting Secondary Student Academic Performance: Empirical Analysis and Linear Regression Modeling on Final Mathematics Examination Grades**

---

## 2. Problem Definition
Academic achievement in secondary school mathematics serves as a foundational gateway for university matriculation and technical career trajectories. Early identification of students at risk of underachievement enables educational institutions to implement targeted academic interventions, tutoring, and socio-emotional support systems before high-stakes final examinations.

The regression problem investigated in this project is the prediction of a student's final examination grade ($G3$, scored on a standard continuous Portuguese grading scale from $0$ to $20$) utilizing socio-demographic, family background, and academic behavioral indicators.

Formally, given a feature vector of student characteristics:

$$\mathbf{x} = [x_{\text{studytime}}, x_{\text{failures}}, x_{\text{absences}}, x_{\text{Medu}}, x_{\text{Fedu}}, x_{\text{higher}}]^T \in \mathbb{R}^p$$

the goal is to learn a mapping function $f: \mathbb{R}^p \to \mathbb{R}$ that predicts the continuous target grade:

$$\hat{y} = f(\mathbf{x}) \approx y \in [0, 20]$$

while minimizing quadratic prediction loss on unseen students.

### Predictor Governance and Methodological Guardrails:
- **Exclusion of Intermediate Exam Grades ($G1$ and $G2$):** First period ($G1$) and second period ($G2$) exam grades are strictly **excluded** from the primary feature set. Because $G1$ and $G2$ correlate at $r > 0.80 - 0.90$ with final grade $G3$, including them transforms the exercise into a trivial auto-regressive extrapolation. Excluding them forces the model to uncover genuine behavioral and socio-academic signals (e.g., weekly study time, past failures, parental education, aspirations).
- **Causality Caution:** Correlation does **not** imply causation. We cannot claim that forcing a student to sit with a textbook for more hours will automatically cause an increase in exam performance; study efficiency, prior foundational aptitude, mental health, and teaching quality act as latent confounders.

---

## 3. Objectives and Success Criteria

### Primary Objectives:
1. Conduct extensive Exploratory Data Analysis (Univariate, Bivariate, and Multivariate) on the UCI Student Performance dataset to uncover distributions, anomalies (e.g., zero-inflation), and correlations with final grade $G3$.
2. Implement a leak-free 60/20/20 train/validation/test split ($237$ Train, $79$ Validation, $79$ Test).
3. Formulate and compare three distinct predictor configurations on the validation set to test whether study time alone or an expanded socio-academic profile minimizes generalization error.
4. Train an Ordinary Least Squares (OLS) Linear Regression model, interpret the fitted parameters ($\beta_0, \beta_j$), and validate statistical regression assumptions through residual diagnostics.
5. Evaluate final predictive performance on held-out test data using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).
6. Serialize the fitted pipeline using `joblib` and provide an interactive prediction utility for prospective student evaluations.

### Quantitative Success Criteria:
- **Prediction Error:** Achieve an RMSE lower than the baseline standard deviation of grades ($\sigma_{y} \approx 4.58$), confirming that the model provides predictive utility superior to predicting the sample mean.
- **Validation-Driven Selection:** Select the configuration that minimizes Validation RMSE without unnecessary complexity.
- **Honest Model Interpretation:** Acknowledge and quantify the fundamental ceiling of predicting human academic performance from static survey data.

---

## 4. Dataset Description and Source
- **Dataset Name:** Student Performance Data Set (Mathematics course, `student-mat.csv`).
- **Source:** Hosted on the official University of California, Irvine (UCI) Machine Learning Repository.
- **Original Citation:** Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal. EUROSIS-ETI.
- **Population:** Secondary school students from two Portuguese public schools: Gabriel Pereira (GP) and Mousinho da Silveira (MS).
- **Dimensionality:** 395 student observations across 33 variables (30 demographic, social, and school-related attributes, plus two intermediate grades $G1, G2$, and one final grade $G3$).

---

## 5. Data Collection
The dataset was ingested programmatically from the UCI repository and loaded with a semicolon delimiter (`;`) into a structured Pandas DataFrame. The primary features investigated in this study include:

| Attribute Name | Variable Category | Data Type | Description / Survey Coding | Scale Range |
| :--- | :--- | :--- | :--- | :--- |
| `studytime` | Academic Behavior | Ordinal / Numeric | Weekly study duration: $1: <2\text{ hrs}, 2: 2-5\text{ hrs}, 3: 5-10\text{ hrs}, 4: >10\text{ hrs}$ | $1 - 4$ |
| `failures` | Academic History | Discrete Numeric | Number of past academic course failures | $0 - 3$ ($4$ if $\ge 4$) |
| `absences` | Academic Engagement | Continuous Numeric | Total recorded school absences throughout the academic year | $0 - 93$ days |
| `Medu` | Family Background | Ordinal / Numeric | Mother's education: $0:\text{none}, 1:\text{4th grade}, 2:\text{5-9th}, 3:\text{secondary}, 4:\text{higher}$ | $0 - 4$ |
| `Fedu` | Family Background | Ordinal / Numeric | Father's education: $0:\text{none}, 1:\text{4th grade}, 2:\text{5-9th}, 3:\text{secondary}, 4:\text{higher}$ | $0 - 4$ |
| `higher` | Student Aspiration | Binary Categorical | Student desires to pursue higher tertiary education: `yes` / `no` | $\{0, 1\}$ |
| **`G3` (Target)** | **Final Outcome** | **Continuous Numeric** | **Final mathematics course grade awarded at the end of the 3rd period** | **$0 - 20$ points** |

---

## 6. Data Exploration
A comprehensive exploratory check on the dataset revealed:
1. **Missing Values:** There are **zero missing or null values** across all 395 rows and 33 attributes.
2. **Duplicate Records:** Zero duplicate rows were found.
3. **Target ($G3$) Characteristics:**
   - Sample Size: $N = 395$
   - Mean: $10.415$ points
   - Standard Deviation: $4.581$ points
   - Minimum: $0.0$ points
   - First Quartile ($25\%$): $8.0$ points
   - Median ($50\%$): $11.0$ points
   - Third Quartile ($75\%$): $14.0$ points
   - Maximum: $20.0$ points
   - Skewness: $-0.733$ (negative skew due to zero-inflation)
4. **The "Zero-Inflation" Phenomenon:** Exactly **38 students ($9.62\%$)** received a final grade of exactly $0$. In the Portuguese secondary school system, a score of 0 does not typically reflect that a student sat for the final exam and answered every question incorrectly; rather, it denotes that the student dropped out mid-year, exceeded the legal absence threshold, or was disqualified from sitting the exam. This represents a known structural anomaly in educational tabular data.

---

## 7. Univariate Analysis
Univariate analysis examines individual variables in isolation to characterize their central tendencies, dispersions, and underlying shapes.

### 7.1 Target Variable Distribution ($G3$)
- **Reference Figure:** [01_target_distribution_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/01_target_distribution_g3.png)
- **Observations:**
  - For students who actively completed the course ($G3 > 0$), grades follow an approximately Gaussian bell curve centered at a mean of approximately $11.5$ points, with the passing threshold established at $10.0$ points.
  - The distribution features a pronounced spike at $G3 = 0$ ($n = 38$), creating a bimodal hybrid mixture: a continuous normal distribution combined with a discrete zero-point mass.

### 7.2 Univariate Distributions of Key Predictors
- **Reference Figure:** [02_univariate_predictors_hist_box.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/02_univariate_predictors_hist_box.png)
- **Observations:**
  - `studytime`: Highly right-skewed. The vast majority of students report studying less than 5 hours per week ($26.6\%$ study $<2$ hrs; $50.1\%$ study $2-5$ hrs; only $6.8\%$ study $>10$ hrs).
  - `failures`: Heavily zero-inflated ($79.0\%$ of students have 0 past failures; $12.7\%$ have 1 failure; $4.3\%$ have 2; $4.1\%$ have 3).
  - `absences`: Positively skewed with a long tail extending up to 75 absences. The median is 4 days, but several students show extreme absenteeism.
  - `Medu` & `Fedu`: Distributed across all levels, with the largest concentration at level 4 (Higher Education: $33.2\%$ for mothers, $24.3\%$ for fathers).

---

## 8. Bivariate Analysis
Bivariate analysis explores the joint relationship and correlation between each predictor and final mathematics grade $G3$.

### 8.1 Study Time vs. Final Grade ($G3$)
- **Reference Figure:** [03_bivariate_studytime_vs_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/03_bivariate_studytime_vs_g3.png)
- **Observations:**
  - Mean grade increases modestly with study time:
    - $<2$ hours/week: Mean $G3 = 10.05$
    - $2-5$ hours/week: Mean $G3 = 10.17$
    - $5-10$ hours/week: Mean $G3 = 11.40$
    - $>10$ hours/week: Mean $G3 = 11.26$
  - Pearson correlation is positive but weak: **$r = +0.0978$**.
  - High study time does **not** guarantee top marks; multiple students studying $>10$ hours still scored $<10$ points, while several students studying $<2$ hours scored $\ge 15$ points.

### 8.2 Past Failures and Parental Education vs. Final Grade ($G3$)
- **Reference Figure:** [04_bivariate_failures_and_medu_vs_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/04_bivariate_failures_and_medu_vs_g3.png)
- **Observations:**
  - `failures` vs $G3$: Shows a substantial negative linear association (**$r = -0.3604$**). Students with 0 past failures average $11.26$ points, whereas students with 3 past failures average only $5.69$ points. Past academic failure is the single strongest negative predictor of future exam performance.
  - `Medu` vs $G3$: Displays a positive linear trend (**$r = +0.2171$**). Students whose mothers attained higher education average $12.10$ points, compared to $8.67$ for mothers with primary education.

### Correlation Summary with $G3$:
| Predictor | Pearson $r$ | Nature of Relationship |
| :--- | :---: | :--- |
| `failures` | **$-0.3604$** | Moderate negative association (past academic distress) |
| `Medu` | **$+0.2171$** | Mild positive association (socio-economic / home academic support) |
| `Fedu` | **$+0.1525$** | Mild positive association |
| `studytime` | **$+0.0978$** | Weak positive association |
| `absences` | **$+0.0342$** | Negligible linear correlation |

---

## 9. Multivariate Analysis
Multivariate analysis investigates joint interactions among three or more predictors simultaneously to assess multi-collinearity and combined predictive power.

### 9.1 Multivariate Correlation Heatmap
- **Reference Figure:** [05_multivariate_correlation_heatmap.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/05_multivariate_correlation_heatmap.png)
- **Observations:**
  - High inter-parental correlation: Mother's education (`Medu`) and Father's education (`Fedu`) correlate strongly at **$r = +0.623$**, reflecting educational assortative mating. In a multi-variable linear model, this can induce mild variance inflation.
  - `failures` negatively correlates with both parental education ($r \approx -0.24$) and `studytime` ($r \approx -0.17$), showing that academic struggle often co-occurs with reduced study habits and lower home educational capital.

### 9.2 Multivariate Pair Plot
- **Reference Figure:** [06_multivariate_pairplot.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/06_multivariate_pairplot.png)
- **Observations:**
  - Multi-feature regression planes demonstrate that no single variable determines $G3$. Instead, high performance requires a confluence of multiple favorable factors: zero past failures, regular attendance, positive educational aspirations, and consistent study time.

---

## 10. Data Cleaning and Preparation
Preprocessing was implemented without data leakage:
1. **Handling Missing Data:** Confirmed complete dataset ($0$ missing values).
2. **Exclusion of Leaky Features:** Intermediate exam grades $G1$ and $G2$ were permanently removed from the predictor pool.
3. **Categorical Encoding:** The binary feature `higher` ("wants to pursue higher education") was encoded numerically using an indicator mapping:

$$\text{higher\_encoded} = \begin{cases} 1 & \text{if } \text{higher} = \text{'yes'} \\ 0 & \text{if } \text{higher} = \text{'no'} \end{cases}$$

4. **Data Partitioning:** All transformations were fitted solely on training splits to prevent data snooping.

---

## 11. Feature Selection and Engineering
Rather than throwing all 32 raw survey features into the regression, which would lead to high variance, multi-collinearity, and loss of interpretability, we formulated a disciplined feature selection strategy:
1. **Behavioral Core:** `studytime` (the primary focus of the assignment) and `absences` (direct measure of engagement).
2. **Academic History:** `failures` (historical baseline of subject-matter aptitude).
3. **Socio-Cultural Capital:** `Medu` and `Fedu` (parental educational environment).
4. **Student Motivation:** `higher_encoded` (intrinsic desire for future educational achievement).

---

## 12. Training, Validation, and Test Splitting
A leak-free, reproducible 60/20/20 split was executed with `random_state=42`:
- **Training Set (60%):** $237$ students (used solely for parameter estimation).
- **Validation Set (20%):** $79$ students (used strictly for model configuration selection).
- **Test Set (20%):** $79$ students (held-out for final evaluation).

| Partition | Proportion | Sample Count | Function in Lifecycle |
| :--- | :---: | :---: | :--- |
| **Training** | 60% | 237 | Fitting ordinary least squares $\boldsymbol{\beta}$ weights |
| **Validation** | 20% | 79 | Selecting the best predictor configuration |
| **Test** | 20% | 79 | Measuring final unbiased generalization error |

---

## 13. Algorithm Selection
The model selected is **Ordinary Least Squares (OLS) Multiple Linear Regression** (`LinearRegression` from `scikit-learn`).

### Mathematical Formulation:
Linear regression models the conditional expectation of target $Y$ as a linear combination of input predictors:

$$\hat{y}_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij}$$

The parameter vector $\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_p]^T$ is estimated by minimizing the Residual Sum of Squares (RSS):

$$\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$

via the closed-form normal equation:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Pedagogical Justification:
- **Direct Interpretability:** Every slope coefficient $\beta_j$ has an unambiguous physical meaning: it represents the expected change in final grade points for a one-unit change in predictor $x_j$, holding all other predictors constant.
- **Benchmark Simplicity:** Provides the fundamental statistical baseline against which any complex non-linear or ensemble technique must be compared.

---

## 14. Model Development and Training
Linear regression models were fitted using OLS on the 237 training instances across candidate predictor sets.

---

## 15. Hyperparameter Tuning and Model Selection
In linear regression without regularization, model selection focuses on choosing the optimal predictor subset. Three candidate configurations were evaluated on the independent 79-sample Validation Set:

- **Config 1 (Study Time Only):** Univariate baseline testing the naive assumption that study time alone explains grades.
- **Config 2 (Academic & Attendance Profile):** `['studytime', 'failures', 'absences']`.
- **Config 3 (Expanded Socio-Academic Profile):** `['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'higher_encoded']`.

### Empirical Validation Comparison:
- **Reference Figure:** [07_model_comparison_validation.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/07_model_comparison_validation.png)
- **Validation Results Table:**

| Model Configuration | Predictors Included | Val MAE | Val RMSE | Val $R^2$ | Train RMSE | Train $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (Study Time Only)** | `['studytime']` | 3.3177 | 4.2707 | -0.0015 | 4.4818 | 0.0077 |
| **Config 2 (Academic & Attendance)**| `['studytime', 'failures', 'absences']` | 3.2961 | 4.1935 | +0.0344 | 4.2041 | 0.1268 |
| **Config 3 (Expanded Profile)** | `['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'higher_encoded']` | **3.2754** | **4.1490** | **+0.0548** | **4.1708** | **0.1406** |

### Critical Educational Findings:
1. **The Failure of the Univariate Study Time Hypothesis:** Config 1 produces a negative validation $R^2$ ($-0.0015$), meaning that predicting a student's grade using study time alone performs slightly *worse* than simply predicting the historical mean grade! This empirically disproves the common myth that study time is the dominant determinant of academic outcomes.
2. **Incremental Gain:** Adding past academic failures and absences (Config 2) lowers validation RMSE from $4.2707$ to $4.1935$ and pushes $R^2$ positive ($+0.0344$).
3. **Optimal Model:** Config 3 achieves the lowest validation error ($\text{RMSE} = 4.1490$, $\text{MAE} = 3.2754$) and the highest explained variance ($R^2 = +0.0548$). It was therefore selected for final evaluation.

---

## 16. Evaluation Metrics
Continuous regression models are evaluated using four foundational metrics:

1. **Mean Absolute Error (MAE):** Average magnitude of absolute errors in natural grade units:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Mean Squared Error (MSE):** Average of squared prediction errors (penalizes large outliers):
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
3. **Root Mean Squared Error (RMSE):** Square root of MSE, returning error scale to points:
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
4. **Coefficient of Determination ($R^2$):** Proportion of target variance explained by the model:
   $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

---

## 17. Final Test Results
The chosen model (Config 3) was evaluated on the 79 completely unseen test students.

### Computed Test Performance Metrics:
- **Mean Absolute Error (MAE):** **$3.3894$ grade points**
- **Mean Squared Error (MSE):** **$19.6402$**
- **Root Mean Squared Error (RMSE):** **$4.4317$ grade points**
- **Coefficient of Determination ($R^2$):** **$0.2155$ ($21.55\%$ explained variance)**

### Fitted OLS Regression Equation:
$$\hat{G3} = 9.2403 - 2.0051(\text{failures}) + 0.7998(\text{higher\_encoded}) + 0.5859(\text{Medu}) - 0.3976(\text{Fedu}) + 0.2030(\text{studytime}) + 0.0210(\text{absences})$$

### Interpretation of Model Weights:
- **Reference Figure:** [10_regression_coefficients.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/10_regression_coefficients.png)
1. **Past Failures ($\beta = -2.0051$):** By far the most impactful coefficient. For every past course failure on record, a student's expected final exam score drops by approximately **$2.0$ full grade points**, holding all other variables constant.
2. **Higher Education Aspiration ($\beta = +0.7998$):** Intrinsic motivation to attend university yields an expected gain of **$+0.80$ points**.
3. **Mother's Education Level ($\beta = +0.5859$):** Each tier increase in maternal education contributes **$+0.59$ points**.
4. **Weekly Study Time ($\beta = +0.2030$):** Advancing one study time bracket (e.g., from $<2$ hrs to $2-5$ hrs) contributes an expected gain of **$+0.20$ points**.
5. **Absences ($\beta = +0.0210$):** Mild, near-zero positive slope. In the Portuguese dataset, moderate absences frequently occur among high-performing students who skip non-critical lectures to study independently.

---

## 18. Results and Discussion

### Predicted vs. Actual Exam Scores:
- **Reference Figure:** [08_predicted_vs_actual_test.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/08_predicted_vs_actual_test.png)
- **Observations:**
  - Predictions cluster along a linear corridor between grades $7.0$ and $14.0$.
  - The model effectively predicts passing versus failing tendencies for regular students, but struggles on the extreme tails (students scoring $0$ or $19-20$).

### Residual Diagnostics:
- **Reference Figure:** [09_residual_plot_test.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/09_residual_plot_test.png)
- **Residual Findings:**
  - The residual histogram displays an approximately normal bell shape centered at zero, satisfying the linear regression error normality assumption for mainstream students.
  - However, a prominent negative residual cluster exists on the left side of the scatter plot. These correspond to students who received a grade of $0$. Because their predictors indicated expected grades of $4 - 9$, the model produces large negative residuals ($e_i = 0 - \hat{y}_i \approx -6$).

---

## 19. Overfitting and Underfitting Considerations
- **Underfitting in Config 1:** With only `studytime`, the model severely underfits ($\text{Train } R^2 = 0.0077$, $\text{Val } R^2 = -0.0015$). The hypothesis space is too constrained to capture student performance variance.
- **Generalization Stability in Config 3:**
  - Train RMSE: $4.1708$, Train $R^2 = 0.1406$
  - Val RMSE: $4.1490$, Val $R^2 = 0.0548$
  - Test RMSE: $4.4317$, Test $R^2 = 0.2155$
  The consistency of RMSE ($4.15 - 4.43$ points) across all three partitions proves that the linear model does not overfit. It captures the true linear signal without memorizing training idiosyncrasies.

---

## 20. Prediction Demonstration
The serialized model artifact (`linear_regression_student.joblib`) was loaded and tested across three distinct student archetypes:

```text
--- Scenario A: High-Engagement Student ---
Profile: Daily study (5-10 hrs/wk), 0 past failures, 2 absences, mother has higher ed, plans higher ed
Inputs: {'studytime': 3, 'failures': 0, 'absences': 2, 'Medu': 4, 'Fedu': 4, 'higher': 'yes'}
Predicted Grade (G3):     11.44 / 20 (Pass)

--- Scenario B: Academic-Risk Student ---
Profile: Minimal study (<2 hrs/wk), 2 past failures, 12 absences, basic parental education, no higher ed plans
Inputs: {'studytime': 1, 'failures': 2, 'absences': 12, 'Medu': 1, 'Fedu': 1, 'higher': 'no'}
Predicted Grade (G3):     5.87 / 20 (Fail / At-Risk)

--- Scenario C: Real Unseen Test Sample ---
Profile: Actual student from held-out test set
Inputs: studytime=1, failures=3, absences=0, Medu=3, Fedu=2, higher='no'
Actual Exam Grade (G3):    0.0 / 20
Predicted Grade (G3):     4.39 / 20 (Fail / At-Risk)
Absolute Prediction Error: 4.39 grade points
```

The model successfully flags Scenario B and C as high academic risks ($G3 < 10.0$) and Scenario A as on-track for passing ($G3 \ge 10.0$).

---

## 21. Deployment Approach
The fitted regression model and its preprocessing dictionary were serialized to `Regression/models/linear_regression_student.joblib` ($1.44\text{ KB}$).

### Production Deployment Architecture:
1. **School Early-Warning System (EWS):** Can be integrated into a student information system (SIS) dashboard. At the start of the academic year, counselors enter background attributes to calculate risk scores.
2. **Automated Triage:** Students with $\hat{G3} < 10.0$ are automatically flagged for supplemental tutoring and mentor pairing.
3. *Honesty Note:* This project provides a verified offline serialized artifact and programmatic CLI inference function. Cloud deployment has not been executed.

---

## 22. Limitations
1. **The Predictability Ceiling of Human Academic Outcomes:** An $R^2$ of $0.2155$ indicates that roughly $78\%$ of exam grade variance remains unexplained by the six predictors. Complex human learning is governed by unmeasured latent factors: student psychological well-being, curriculum difficulty, teacher pedagogy, test anxiety, and peer dynamics.
2. **Zero-Inflation Truncation:** Standard linear regression assumes unbounded continuous outputs $\hat{y} \in (-\infty, +\infty)$ and cannot natively handle the structural zeroes produced by course dropouts. A two-stage hurdle model (logistic regression for dropout classification followed by truncated regression for active test-takers) would be required to handle this phenomenon fully.
3. **Linearity Assumption:** OLS enforces linear relationships, failing to account for diminishing marginal returns (e.g., studying 15 hours versus 12 hours yields less incremental gain than studying 3 hours versus 0 hours).

---

## 23. Conclusion
Project B successfully designed, trained, and verified an Ordinary Least Squares Linear Regression pipeline on the UCI Student Performance dataset. Intermediate exam grades were excluded to enforce honest socio-academic learning. Through validation-guided model selection, we demonstrated that while weekly study time alone explains virtually none of final grade variance, an expanded profile incorporating past academic failures, attendance, and parental education achieves an $R^2$ of $21.55\%$ on unseen test data with an MAE of $3.39$ points. Past academic failure was identified as the single largest negative determinant of future exam scores ($\beta = -2.01$).

---

## 24. References
1. Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In Proceedings of the 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5–12.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.
3. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

---

# Comparative Analysis Section: Univariate vs. Bivariate vs. Multivariate Analysis

Exploratory Data Analysis (EDA) is the cornerstone of every machine learning workflow. In both classification and regression, statistical depth scales with the number of variables analyzed simultaneously:

### 1. Univariate Analysis
- **Definition:** Statistical and visual examination of a single variable in isolation.
- **Number of Variables:** Exactly 1 ($X_i$ or target $Y$).
- **Purpose:** Characterize the underlying probability distribution, identify central tendency (mean, median), measure spread (variance, standard deviation, interquartile range), inspect skewness, and detect anomalies or zero-inflation.
- **Techniques & Visualizations:** Histograms with KDE curves, Box plots, Summary statistics tables, Bar charts.
- **Application in Project B:**
  - Distribution of target grade $G3$ showing normality among test-takers alongside a $9.6\%$ zero-inflation cluster ([01_target_distribution_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/01_target_distribution_g3.png)).
  - Distribution of study time showing heavy positive skewness ([02_univariate_predictors_hist_box.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/02_univariate_predictors_hist_box.png)).
- **Strengths:** Intuitive, computationally minimal, exposes data hygiene flaws and distributional anomalies early.
- **Limitations:** Completely blind to dependencies, correlations, and interactions between features.

### 2. Bivariate Analysis
- **Definition:** The joint statistical and visual examination of the relationship between two variables.
- **Number of Variables:** Exactly 2 ($X_i$ and $X_j$, or $X_i$ and target $Y$).
- **Purpose:** Determine empirical correlation, assess linear and monotonic associations, evaluate group differences, and test whether an individual predictor holds predictive association with the target.
- **Techniques & Visualizations:** Scatter plots with trendlines, Grouped box plots, Pearson and Spearman correlation coefficients.
- **Application in Project B:**
  - Scatter plot of `studytime` vs $G3$ showing a weak positive relationship ($r = +0.0978$) ([03_bivariate_studytime_vs_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/03_bivariate_studytime_vs_g3.png)).
  - Grouped boxplots of `failures` vs $G3$ revealing a steep downward slope ($r = -0.3604$) ([04_bivariate_failures_and_medu_vs_g3.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/04_bivariate_failures_and_medu_vs_g3.png)).
- **Strengths:** Pinpoints primary drivers of the target variable and tests direct hypotheses.
- **Limitations:** Susceptible to confounding variables; cannot capture non-linear joint interactions or multi-collinearity among multiple features.

### 3. Multivariate Analysis
- **Definition:** The simultaneous statistical and visual examination of three or more variables.
- **Number of Variables:** $\ge 3$ ($X_1, X_2, \dots, X_p, Y$).
- **Purpose:** Uncover complex multi-feature interactions, detect multi-collinearity, analyze covariance structure, and evaluate how features combine to predict the target.
- **Techniques & Visualizations:** Pair plot grids, Correlation heatmaps with clustering, Variance Inflation Factor (VIF), Multiple regression weight analysis.
- **Application in Project B:**
  - Multivariate correlation heatmap showing high inter-parental correlation ($r = +0.623$ between `Medu` and `Fedu`) and joint relationships with $G3$ ([05_multivariate_correlation_heatmap.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/05_multivariate_correlation_heatmap.png)).
  - Multivariate pairplot displaying regression planes across predictors ([06_multivariate_pairplot.png](file:///home/yilu/Documents/ML_Assignment/Regression/figures/06_multivariate_pairplot.png)).
- **Strengths:** Reflects real-world educational complexity where outcomes result from a confluence of multiple factors.
- **Limitations:** Higher computational cost; difficult to visualize directly beyond three dimensions without dimensionality reduction.

---

### Comparative Analysis Table Across Project A (Classification) and Project B (Regression)

| Comparative Dimension | Univariate Analysis | Bivariate Analysis | Multivariate Analysis |
| :--- | :--- | :--- | :--- |
| **Number of Variables** | 1 variable ($X$) | 2 variables ($X_1, X_2$ or $X, Y$) | 3 or more variables ($X_1, \dots, X_p, Y$) |
| **Primary Objective** | Characterize individual distribution, central tendency, and detect anomalies | Identify pairwise association, correlation, and direct discriminatory power | Identify higher-order interactions, multi-collinearity, and latent structure |
| **Common Visualizations** | Histograms, KDE plots, Box plots, Bar charts | Scatter plots, Grouped box plots, Scatter with class hue | Pair plot matrices, Correlation heatmaps, 3D scatter, PCA biplots |
| **Mathematical Tools** | Mean, Median, Variance, Std Dev, IQR, Skewness | Pearson $r$, Spearman $\rho$, Covariance, ANOVA $F$-test | Covariance matrix, Eigenvalues/Eigenvectors, Multi-collinearity (VIF) |
| **Application in Project A (Iris Classification)** | Histograms of petal/sepal dimensions; species count balance check (50 each) | Scatter of Petal Length vs. Petal Width; grouped boxplots by species | $4 \times 4$ Pair plot with species hue; complete correlation heatmap; tree feature importance |
| **Application in Project B (Student Exam Regression)** | Histogram and box plot of final grade $G3$, study time distribution | Scatter plot of Study Time vs. $G3$; box plots of parental education vs. $G3$ | Correlation matrix heatmap of all numerical predictors; multi-feature regression plane |
| **Key Insights Revealed (Project A)** | Petal measurements are distinctly bimodal; no missing values; balanced classes | Petal length and width correlate at $r=0.963$; *Setosa* is linearly separable | Petal features dominate Gini importance ($94.8\%$); sepal features are redundant |
| **Key Insights Revealed (Project B)** | Final exam grade $G3$ is approximately normal but has a zero-inflation peak (unattempted exams) | Weak-to-moderate positive correlation between study time and $G3$; strong negative correlation with past failures | Multiple non-academic factors (study time, failures, absences, school support) jointly predict final performance |
| **Main Limitation** | Ignores all relationships with other variables | Vulnerable to confounding variables and Simpson's Paradox | Difficult to visualize directly without projection techniques |

