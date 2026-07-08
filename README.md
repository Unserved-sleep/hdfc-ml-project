# HDFC ML Project

An end-to-end machine learning project covering supervised learning, semi-supervised learning, and unsupervised clustering — applied to HDFC loan data and the UCI Bank Marketing dataset.

---

## Project Structure

```
hdfc-ml-project/
│
├── notebooks/                        # Jupyter notebooks (numbered in order)
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_loan_approval_prediction.ipynb
│   ├── 03_default_risk_analysis.ipynb
│   ├── 04_loan_amount_prediction.ipynb
│   ├── 05_svm_analysis.ipynb
│   ├── 06_knn_analysis.ipynb
│   ├── 07_unsupervised_learning.ipynb
│   └── 08_advanced_clustering.ipynb
│
├── src/                              # Shared utility modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── dataset/
│   ├── hdfc_loan_dataset_full_enriched.csv
│   ├── hdfc_loan_dataset_clustered.csv   # HDFC dataset with cluster labels (legacy)
│   └── bank.csv                          # UCI Bank Marketing dataset
│
├── models/                           # Saved trained models (.pkl)
│   ├── loan_approval_model.pkl
│   ├── risk_model.pkl
│   ├── loan_amount_model.pkl
│   └── svm_model.pkl
│
├── visuals/                          # Saved plots and charts
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 — Exploratory Data Analysis
Comprehensive analysis of the HDFC loan dataset.
- Distribution plots for income, loan amount, CIBIL score
- Loan status breakdown by occupation, property area, state
- Correlation heatmap
- Outlier detection and missing value analysis

### 02 — Loan Approval Prediction
Binary classification: predict whether a loan application is **Approved** or **Rejected**.
- Models: Logistic Regression → Decision Tree → Random Forest (GridSearchCV)
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC curve
- Feature importance via Random Forest
- Best model saved to `models/loan_approval_model.pkl`

### 03 — Default Risk Analysis
Binary classification: identify **high-risk** applicants likely to default.
- Target engineered from `Default_History_Count`
- Class imbalance handled with **SMOTE**
- Models: Logistic Regression → Random Forest (GridSearchCV) → Gradient Boosting
- Evaluation: ROC-AUC scoring, classification reports, feature importance
- Best model saved to `models/risk_model.pkl`

### 04 — Loan Amount Prediction
Regression task: predict the **loan amount** an applicant will receive.
- Models: Linear Regression → Random Forest Regressor (GridSearchCV) → Gradient Boosting Regressor
- Evaluation: MAE, MSE, RMSE, R²
- Residual plot and actual vs predicted scatter plot
- Best model saved to `models/loan_amount_model.pkl`

### 05 — SVM Analysis
Binary classification using **Support Vector Machines**.
- Kernels compared: Linear vs RBF (baseline)
- Hyperparameter tuning: `C`, `gamma`, `kernel` via GridSearchCV (scoring: ROC-AUC)
- Evaluation: accuracy, precision, recall, F1, ROC curve, confusion matrix
- Linear SVM coefficients extracted for feature importance interpretation
- Best model saved to `models/svm_model.pkl`

### 06 — KNN Analysis
Binary classification using **K-Nearest Neighbours (KNN)** on the HDFC loan dataset.
- Optimal `k` found by plotting train vs validation accuracy across k=1..30 (bias-variance tradeoff)
- Compared **uniform** vs **distance-weighted** voting
- Decision boundary visualised in 2D PCA space
- Evaluation: accuracy, precision, recall, F1, ROC curve, confusion matrix
- Final comparison against Logistic Regression and Random Forest baselines

### 07 — Unsupervised Learning — Term Deposit Subscription
Unsupervised clustering on the **UCI Bank Marketing dataset** to predict term deposit subscription behaviour.
- Feature engineering: `was_previously_contacted`, `balance_per_age`, `campaign_contact_rate`
- Encoding → Scaling → **PCA** (95% variance retained) → K-Means
- Elbow Method + Silhouette Score for optimal k
- Cluster visualisation in 2D PCA space
- Cluster profiling: numeric heatmap + categorical breakdowns
- Subscription rate analysis per cluster with business interpretation

### 08 — Advanced Clustering (K-Means + K-Modes + DBSCAN)
Three complementary clustering algorithms on the **UCI Bank Marketing dataset**.

| Algorithm | Input | Purpose |
|-----------|-------|---------|
| **K-Means** | Numerical (scaled) | Segment by financial behaviour |
| **K-Modes** | Categorical (raw) | Segment by demographics |
| **DBSCAN** | Numerical (scaled) | Detect outlier / anomalous customers |

- K-Modes elbow on cost, mode centres extracted per cluster
- DBSCAN eps estimated via k-NN distance plot
- Outlier profile vs normal customer comparison
- Cross-algorithm comparison table (ARI, cluster count, outliers)
- All three cluster labels saved to `dataset/bank_clustered.csv`

---

## Source Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `preprocessing.py` | Load dataset, clean, impute, scale, one-hot encode via ColumnTransformer |
| `feature_engineering.py` | Domain features: Total_Income, EMI_Income_Ratio, Loan_Income_Ratio, etc. |
| `train.py` | Build sklearn Pipeline, run GridSearchCV, generate predictions |
| `evaluate.py` | Classification and regression metrics, confusion matrix, ROC curve, feature importance |
| `utils.py` | Save/load models (joblib), plot style, model comparison, DataFrame summary |

---

## Datasets

| File | Description |
|------|-------------|
| `hdfc_loan_dataset_full_enriched.csv` | 1,000 HDFC loan applications with 47 features |
| `hdfc_loan_dataset_clustered.csv` | Same dataset with `KMeans_Cluster` and `KMedoids_Cluster` columns added (legacy) |
| `bank.csv` | UCI Bank Marketing dataset (4,521 records, 17 features, `;` separated) |

---

## Models

| File | Algorithm | Task |
|------|-----------|------|
| `loan_approval_model.pkl` | Random Forest | Loan approval classification |
| `risk_model.pkl` | Random Forest | Default risk classification |
| `loan_amount_model.pkl` | Random Forest | Loan amount regression |
| `svm_model.pkl` | SVM (tuned) | Loan approval classification |

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
pandas==2.2.3
numpy==2.1.1
scikit-learn==1.7.2
matplotlib==3.10.1
seaborn==0.13.2
joblib==1.5.3
imbalanced-learn==0.14.2
kmodes==0.12.2
```

---

## Branch

Active development branch: `dev`
