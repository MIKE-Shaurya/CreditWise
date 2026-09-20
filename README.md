# CreditWise — Loan Approval Model

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![status](https://img.shields.io/badge/status-active-brightgreen)

A binary classifier that predicts whether a loan application will be
approved, based on applicant financials, credit history, and loan
details.

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Dataset columns](#dataset-columns)
- [Model performance](#model-performance)
- [Making a prediction](#making-a-prediction)
- [Models](#models)

## Project structure

```
CreditWise/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # put loan_approval_data.csv here
│   └── processed/                # cleaned/engineered CSVs land here
├── src/
│   ├── preprocessing.py          # load, clean, impute, encode
│   ├── feature_engineering.py    # engineered features
│   └── modeling.py               # train/test split, models, metrics
└── notebooks/
    ├── 01_data_preprocessing.ipynb
    ├── 02_eda.ipynb
    ├── 03_feature_engineering.ipynb
    └── 04_modeling.ipynb
```

The notebooks are thin — all real logic lives in `src/` and is
imported, so the same functions are testable and reusable outside a
notebook. Each notebook saves its output CSV to `data/processed/` for
the next one to read, so you can run them in order (`01` → `02` → `03`
→ `04`).

## Setup

```bash
git clone <your-repo-url>
cd CreditWise
pip install -r requirements.txt
```

Place your `loan_approval_data.csv` in `data/raw/`, then run the
notebooks in order.

## Dataset columns

| Column | Description |
|---|---|
| `Applicant_ID` | Unique ID (dropped before modeling) |
| `Applicant_Income`, `Coapplicant_Income` | Monthly income |
| `Employment_Status` | Salaried / Self-employed / Unemployed |
| `Age`, `Dependents`, `Marital_Status`, `Education_Level`, `Gender` | Demographics |
| `Credit_Score` | 550–800 credit score |
| `Existing_Loans`, `DTI_Ratio` | Existing debt load |
| `Savings`, `Collateral_Value` | Assets |
| `Loan_Amount`, `Loan_Term`, `Loan_Purpose` | Requested loan |
| `Property_Area`, `Employer_Category` | Additional context |
| `Loan_Approved` | Target (Yes/No) |

## Model performance

Results on a held-out 20% test split (`random_state=42`):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |
| KNN | 0.760 | 0.627 | 0.525 | 0.571 |

Logistic Regression and Naive Bayes are the strongest baselines here,
both landing at 86.5% accuracy. Re-run `04_modeling.ipynb` after
updating the dataset or the engineered features to refresh this table.

## Making a prediction

```python
import sys
sys.path.append(".")

from src.preprocessing import clean_and_encode
from src.feature_engineering import engineer_features
from src.modeling import split_and_scale
from sklearn.linear_model import LogisticRegression

# Build the model-ready dataset
df = clean_and_encode("data/raw/loan_approval_data.csv")
df = engineer_features(df)

X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the held-out test rows
predictions = model.predict(X_test)
print(predictions[:10])   # 1 = approved, 0 = not approved

# Predict on a single new applicant:
# new_row = scaler.transform([[...same feature order as X_train...]])
# model.predict(new_row)
```

`Loan_Approved` is label-encoded during preprocessing, so predictions
come back as `1` (approved) / `0` (not approved) rather than `Yes`/`No`.

## Models

Three baseline classifiers are compared: Logistic Regression, KNN, and
Gaussian Naive Bayes, using precision, recall, F1, accuracy, and the
confusion matrix.

## License

Add a license of your choice (MIT is a common default for portfolio projects).
