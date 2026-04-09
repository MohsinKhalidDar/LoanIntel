# 🏦 LoanIntel — Loan Approval Prediction System

> An intelligent, ML-powered system that predicts whether a bank loan application should be approved or rejected based on applicant financial and demographic data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loanintel.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Problem Statement

Bank receives hundreds of loan applications daily. Manual verification is slow, inconsistent, and prone to bias — leading to two business problems:
- ❌ Good customers get rejected → loss of business
- ❌ Risky customers get approved → financial losses

**LoanIntel** solves this by automating the decision using historical data and machine learning.

---

## 🎯 Project Highlights

| Item | Detail |
|------|--------|
| **Type** | Binary Classification |
| **Dataset** | 1000 applicants, 19 features |
| **Target** | `Loan_Approved` (Yes / No) |
| **Models** | Logistic Regression, KNN, Naive Bayes |
| **Best Model** | Naive Bayes (highest Precision) |
| **Deployment** | Streamlit Web App |

---

## 📊 Model Performance (After Feature Engineering)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 87.5% | 79.0% | 80.3% | 79.7% |
| KNN (k=5) | 75.5% | 62.0% | 50.8% | 55.9% |
| **Naive Bayes** | **86.5%** | **78.3%** | **77.0%** | **77.7%** |

> **Why Precision?** In banking, approving a bad loan (False Positive) is more costly than rejecting a good customer. Precision directly measures how often approved loans are genuinely good.

---

## 🗂️ Dataset Features

- **Personal:** Gender, Age, Marital Status, Dependents, Education Level
- **Financial:** Applicant Income, Co-applicant Income, Savings, DTI Ratio, Credit Score, Existing Loans
- **Loan:** Loan Amount, Loan Term, Loan Purpose, Collateral Value
- **Employment:** Employment Status, Employer Category, Property Area

---

## 🔧 ML Pipeline

```
Raw Data
   ↓
Missing Value Imputation (mean for numerical, mode for categorical)
   ↓
Drop Applicant_ID (identifier — not a feature)
   ↓
Label Encoding  (Education_Level, Loan_Approved)
   ↓
One-Hot Encoding  (Employment_Status, Marital_Status, Loan_Purpose, Property_Area, Gender, Employer_Category)
   ↓
Train-Test Split  (80/20, random_state=42)
   ↓
Feature Engineering  (DTI_Ratio², Credit_Score²)
   ↓
StandardScaler  (fit on train only, transform on test)
   ↓
Model Training & Evaluation
```

---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/MohsinKhalidDa/LoanIntel.git
cd LoanIntel

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
LoanIntel/
├── app.py                    # Streamlit web application
├── loan_approval_data.csv    # Dataset
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore
└── notebooks/
    └── loan_intel.ipynb      # Jupyter notebook (EDA + model building)
```

---

## 💡 Key Learnings

- Handled class imbalance — 65% rejected, 35% approved — using precision as primary metric
- Prevented data leakage by fitting scaler only on training data
- Applied `drop='first'` in OneHotEncoding to avoid the dummy variable trap
- Feature engineering (squaring Credit_Score and DTI_Ratio) improved model sensitivity to non-linear patterns

---

## 🔮 Future Scope

- [ ] Add Random Forest and XGBoost models
- [ ] Apply SMOTE to handle class imbalance
- [ ] Add cross-validation (K-Fold)
- [ ] Integrate SHAP values for prediction explainability
- [ ] Add user authentication for bank officer access

Built with Python, Scikit-learn, and Streamlit
