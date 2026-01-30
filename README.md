# Pima Indians Diabetes Prediction (Machine Learning)

End-to-end machine learning project to predict diabetes outcomes using the **Pima Indians Diabetes Database** (Kaggle / UCI).  
This repository includes **Phase 1 (Data Preparation & Visualisation)** and **Phase 2 (Predictive Modelling)**.

---

## Overview

### Goal
Build and evaluate classification models that predict whether a patient has diabetes (**Outcome = 1**) based on clinical/diagnostic measurements (**Outcome = 0** otherwise).

### Dataset
- **Records:** 768 patients (benchmark dataset)
- **Population notes (dataset description):** Female patients, age 21+, Pima Indian heritage
- **Target:** `Outcome` (binary: 0/1)

### Features (8 predictors)
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

---

## What I Did (Phase 1 → Phase 2)

## Phase 1 — Data Preparation & Visualisation

### Data cleaning & quality handling
- Removed duplicates
- Treated biologically impossible **0 values** as missing for key medical columns:
  - **Mean imputation:** Glucose, BloodPressure
  - **Median imputation (skewed):** SkinThickness, Insulin, BMI
- Applied **rule-based outlier filtering** to reduce extreme values (e.g., unusually high insulin/BMI)

### Exploratory analysis (EDA)
- Summary statistics + distribution checks
- Visual validation of outliers and feature distributions (histograms/boxplots)
- Outcome distribution checks (class balance)

> Note: Row counts after outlier filtering may vary slightly depending on the exact Kaggle export/version used.

---

## Phase 2 — Predictive Modelling

### Train/Test Split
- 80/20 split (`test_size = 0.2`, `random_state = 0`)

### Feature selection
Used **SelectKBest (ANOVA F-test / `f_classif`)** to rank predictive strength.  
Top signals typically include:
- Glucose (strongest)
- BMI
- Age
- Pregnancies  
(+ other features with moderate contribution)

### Models trained + tuned (GridSearchCV)
Trained and evaluated multiple classifiers with hyperparameter tuning:

- **K-Nearest Neighbors (KNN)** — repeated stratified CV (10 folds × 3 repeats), F1 scoring
- **Naive Bayes (GaussianNB)** — tuned `var_smoothing`
- **Support Vector Machine (SVM)** — tuned kernel/C, F1 scoring
- **Decision Tree** — tuned depth/leaf size/criterion
- **Random Forest** — tuned depth/splits/estimators, F1 scoring
- **Logistic Regression** — tuned C/solver

Additionally:
- Trained a **neural network** and compared performance via ROC-AUC
- Used **ROC curves + AUC** for model comparison
- Applied **paired t-tests** to evaluate whether AUC differences were statistically significant

---

## Results Summary (from notebook run)

### Best hyperparameters (high-level)
- **KNN:** metric=manhattan, n_neighbors=17, p=1, weights=distance  
- **Naive Bayes:** var_smoothing=0.01  
- **SVM:** kernel=rbf, C=10, gamma=scale  
- **Decision Tree:** criterion=entropy, max_depth=5, min_samples_leaf=20  
- **Random Forest:** max_depth=10, min_samples_split=10, min_samples_leaf=1, n_estimators=100  
- **Logistic Regression:** C=10, solver=liblinear  

### Test-set performance (key metrics)
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|------|----------:|---------------------:|-----------------:|-------------:|
| KNN | 0.76 | 0.64 | 0.56 | 0.60 |
| Naive Bayes | 0.75 | 0.61 | 0.58 | 0.60 |
| SVM | 0.79 | 0.72 | 0.54 | 0.62 |
| Decision Tree | 0.73 | 0.57 | 0.62 | 0.59 |
| Random Forest | 0.77 | 0.63 | 0.65 | 0.64 |
| **Logistic Regression** | **0.79** | **0.69** | **0.65** | **0.67** |

✅ **Best overall (balanced): Logistic Regression** — highest F1 with strong accuracy and interpretability.

### ROC-AUC comparison (cross-validated)
- Logistic Regression: **~0.83**
- Random Forest: **~0.83**
- Gradient Boosting: **~0.82**
- Decision Tree: **~0.77**
- Neural Network: **~0.64**

**Significance testing (paired t-tests):**  
Logistic Regression significantly outperformed Decision Tree and Neural Net by AUC, while differences vs Random Forest / Gradient Boosting were not statistically significant in this run.

---

## How to Run

### Run notebooks
1. Install dependencies (see below)
2. Open notebooks in `notebooks/`:
   - `Phase1_DataPrep_EDA.ipynb`
   - `Phase2_PredictiveModeling.ipynb`
3. Ensure the dataset path matches the folder layout:
   - If dataset is in `raw_data/diabetes.csv`, use:
     ```python
     data = pd.read_csv("../raw_data/diabetes.csv")
     ```

---

## Install Dependencies
``` bash
pip install -r requirements.txt
```

---

## Repository Contents
- **`raw_data/`** — dataset (or dataset placeholder + download instructions)
- **`notebooks/`** — Phase 1 + Phase 2 notebooks
- **`reports/`** — optional exported HTML/PDF reports
- **`output/`** — optional saved plots, figures, and generated artifacts

---

## Tools & Skills Demonstrated

### Core Stack
- **Python:** pandas, numpy  
- **Visualisation:** matplotlib, seaborn  

### Data Preparation
- Data cleaning & imputation (**mean/median**)
- Outlier handling (**rule-based filtering**)
- EDA + visual validation (histograms/boxplots)

### Modelling
- Feature selection: **SelectKBest / ANOVA F-test**
- Model training & hyperparameter tuning: **GridSearchCV**
- Evaluation: classification report, confusion matrix, precision/recall/F1
- Model comparison: **ROC curves, AUC**, paired significance testing

---

## Data Source
Kaggle / UCI **Pima Indians Diabetes** dataset (commonly attributed to the National Institute of Diabetes and Digestive and Kidney Diseases).

---

## Author
**Akshay Kumar**  
GitHub: https://github.com/akshay040
