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
- [API and web app](#api-and-web-app)
- [Run with Docker](#run-with-docker)

## Project structure

```
CreditWise/
├── README.md
├── requirements.txt              # everything (notebooks + API + UI)
├── requirements-api.txt          # what the API container installs
├── requirements-ui.txt           # what the Streamlit container installs
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.ui
├── data/
│   ├── raw/                     # put loan_approval_data.csv here
│   └── processed/                # cleaned/engineered CSVs land here
├── models/                       # trained model bundle lands here
├── api/
│   └── main.py                   # FastAPI service
├── app/
│   └── streamlit_app.py          # Streamlit front end (calls the API)
├── src/
│   ├── preprocessing.py          # load, clean, impute, encode
│   ├── feature_engineering.py    # engineered features
│   ├── modeling.py               # train/test split, models, metrics
│   ├── train.py                  # train + save the model bundle
│   └── inference.py              # one applicant -> features -> prediction
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

```
git clone <repo-url>
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

## API and web app

The served model is Logistic Regression trained on the engineered
features (`src/train.py`). Training saves one file,
`models/creditwise_model.joblib`, containing the model, the scaler, the
final feature order and the category lists, which is everything the API
needs to score a single new applicant.

Run locally, from the repo root:

```
pip install -r requirements.txt
python -m src.train                          # needs data/raw/loan_approval_data.csv
uvicorn api.main:app --reload                # http://localhost:8000/docs
streamlit run app/streamlit_app.py           # http://localhost:8501
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check, reports whether a model is loaded |
| `GET /options` | Allowed values for each categorical field (from the training data) |
| `GET /model-info` | Model name, training time, test metrics |
| `POST /predict` | Score one application, returns `approved` and `approval_probability` |

Example request:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Applicant_Income": 5000, "Coapplicant_Income": 0, "Age": 35, "Dependents": 1,
  "Credit_Score": 700, "Existing_Loans": 1, "DTI_Ratio": 0.3, "Savings": 10000,
  "Collateral_Value": 20000, "Loan_Amount": 15000, "Loan_Term": 36,
  "Employment_Status": "Salaried", "Marital_Status": "Married",
  "Loan_Purpose": "Personal", "Property_Area": "Urban", "Gender": "Male",
  "Employer_Category": "Private", "Education_Level": "Graduate"
}'
```

Call `GET /options` to see the exact category spellings your dataset
uses; an unknown value returns a 422 listing the allowed ones. The
Streamlit app fills its dropdowns from that same endpoint.

## Run with Docker

The compose file runs two containers, `api` (port 8000) and `ui`
(port 8501), plus a one-off `train` job.

```
# 1. Put loan_approval_data.csv in data/raw/, then train inside Docker
docker compose run --rm train

# 2. Start the API and the web app
docker compose up --build
```

Then open http://localhost:8501 for the app or http://localhost:8000/docs
for the API docs.

Notes:

- The model is mounted into the API container from `./models`, not baked
  into the image. After retraining, the API picks up the new file
  automatically, with no restart needed.
- Train with `docker compose run --rm train` rather than locally. A model
  pickled with one scikit-learn version can misbehave when loaded by
  another, and this way both sides use the same image.
- If you start the stack before training, the app shows a "no trained
  model" message instead of crashing. Train, then refresh the page.


