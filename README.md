# CreditWise — Loan Approval Model

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/)
[![status](https://img.shields.io/badge/status-active-brightgreen)](#)

A binary classifier that predicts whether a loan application will be approved, based on applicant financials, credit history, and loan details. The trained model is served through a FastAPI backend and a Streamlit web app.

## 🚀 Live demo

| | Link |
| --- | --- |
| **Web app (Streamlit)** | https://creditwise-1.streamlit.app/ |
| **API (FastAPI on Render)** | https://creditwise-1.onrender.com |
| **API docs (Swagger)** | https://creditwise-1.onrender.com/docs |

> **Note:** the API runs on Render's free tier, which sleeps after a period of inactivity. The first prediction after a break can take 30–60 seconds while the service wakes up.

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [Dataset columns](#dataset-columns)
- [Model performance](#model-performance)
- [Making a prediction in Python](#making-a-prediction-in-python)
- [Models](#models)
- [Deployment](#deployment)

## Architecture

```
┌──────────────────┐   HTTP (JSON)   ┌──────────────────┐      ┌──────────────┐
│  Streamlit UI    │ ──────────────▶ │  FastAPI service │ ───▶ │ models/      │
│  (Streamlit      │ ◀────────────── │  (Render)        │      │ trained model│
│   Community      │   prediction    │  /predict        │      │ + scaler     │
│   Cloud)         │                 └──────────────────┘      └──────────────┘
└──────────────────┘
```

The UI collects applicant details, sends them to the API, and displays whether the application is likely to be approved along with the approval probability.

## Project structure

```
CreditWise/
├── README.md
├── render.yaml               # Render deployment config for the API
├── requirements.txt          # full environment (notebooks + training)
├── requirements-api.txt      # API-only dependencies
├── requirements-ui.txt       # Streamlit UI-only dependencies
├── api/                      # FastAPI service (serves /predict, /options, ...)
├── app/                      # Streamlit web app
├── models/                   # trained model artifacts loaded by the API
├── data/
│   ├── raw/                  # put loan_approval_data.csv here
│   └── processed/            # cleaned/engineered CSVs land here
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

The notebooks are thin — all real logic lives in `src/` and is imported, so the same functions are testable and reusable outside a notebook. Each notebook saves its output CSV to `data/processed/` for the next one to read, so you can run them in order (`01` → `02` → `03` → `04`).

## Setup

```bash
git clone https://github.com/MIKE-Shaurya/CreditWise.git
cd CreditWise
pip install -r requirements.txt
```

Place your `loan_approval_data.csv` in `data/raw/`, then run the notebooks in order to reproduce the preprocessing, EDA, feature engineering and modeling steps.

## Running locally

Install only what each service needs:

```bash
# API
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000     # adjust module path to match api/

# Streamlit UI (in a second terminal)
pip install -r requirements-ui.txt
streamlit run app/<your_streamlit_file>.py    # adjust to your file name
```

The UI reads the API address from the `API_URL` setting and defaults to `http://localhost:8000`. To point it at the deployed API instead:

```bash
export API_URL="https://creditwise-1.onrender.com"
```

## API reference

Base URL: `https://creditwise-1.onrender.com` — interactive docs at [`/docs`](https://creditwise-1.onrender.com/docs).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `GET` | `/options` | Valid values for the categorical fields |
| `GET` | `/model-info` | Information about the loaded model |
| `POST` | `/predict` | Predict loan approval for one applicant |

### Example request

```bash
curl -X POST 'https://creditwise-1.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "Applicant_Income": 5000,
    "Coapplicant_Income": 0,
    "Age": 35,
    "Dependents": 1,
    "Credit_Score": 700,
    "Existing_Loans": 1,
    "DTI_Ratio": 0.3,
    "Savings": 10000,
    "Collateral_Value": 20000,
    "Loan_Amount": 15000,
    "Loan_Term": 36,
    "Employment_Status": "Salaried",
    "Marital_Status": "Married",
    "Loan_Purpose": "Personal",
    "Property_Area": "Urban",
    "Gender": "Male",
    "Employer_Category": "Private",
    "Education_Level": "Graduate"
  }'
```

### Example response

```json
{
  "approved": false,
  "approval_probability": 0.083,
  "threshold": 0.5
}
```

`approved` is `true` when `approval_probability` is at or above `threshold`. Invalid input returns a `422` validation error describing the offending field.

## Dataset columns

| Column | Description |
| --- | --- |
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
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |
| KNN | 0.760 | 0.627 | 0.525 | 0.571 |

Logistic Regression and Naive Bayes are the strongest baselines here, both landing at 86.5% accuracy. Re-run `04_modeling.ipynb` after updating the dataset or the engineered features to refresh this table.

## Making a prediction in Python

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
```

`Loan_Approved` is label-encoded during preprocessing, so predictions come back as `1` (approved) / `0` (not approved) rather than `Yes`/`No`.

To score a single new applicant, use the web app or the `/predict` endpoint above, which apply the same preprocessing and feature engineering as training.

## Models

Three baseline classifiers are compared: Logistic Regression, KNN, and Gaussian Naive Bayes, using precision, recall, F1, accuracy, and the confusion matrix. The trained model used by the API is stored in `models/`.

## Deployment

| Component | Platform | Notes |
| --- | --- | --- |
| API | [Render](https://render.com/) | Configured in `render.yaml`; installs from `requirements-api.txt` |
| Web app | [Streamlit Community Cloud](https://streamlit.io/cloud) | Main file in `app/`; set the `API_URL` secret to the Render API URL |

To redeploy the UI against a different API, change the `API_URL` secret under **App settings → Secrets** in Streamlit Cloud:

```toml
API_URL = "https://creditwise-1.onrender.com"
```

Keep the scikit-learn version in `requirements-api.txt` identical to the one used to train the model, otherwise the saved model may fail to load.
