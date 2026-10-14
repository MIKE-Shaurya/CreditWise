"""
FastAPI service for the CreditWise loan approval model.

    uvicorn api.main:app --reload

Interactive docs are served at /docs.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import CreditWiseModel, UnknownCategoryError

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/creditwise_model.joblib"))

app = FastAPI(
    title="CreditWise API",
    description="Predict whether a loan application will be approved.",
    version="1.0.0",
)

# The model is loaded on first use and reloaded whenever the file on disk
# changes, so retraining doesn't require restarting the container.
_cache: dict = {"model": None, "mtime": None}


def get_model() -> CreditWiseModel | None:
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except FileNotFoundError:
        return None
    if _cache["model"] is None or _cache["mtime"] != mtime:
        _cache["model"] = CreditWiseModel.load(MODEL_PATH)
        _cache["mtime"] = mtime
    return _cache["model"]


def require_model() -> CreditWiseModel:
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained model found at {MODEL_PATH}. Train one first with "
                "`python -m src.train` (or `docker compose run --rm train`)."
            ),
        )
    return model


class LoanApplication(BaseModel):
    """One loan application. Field names match the columns in the raw dataset."""

    Applicant_Income: float = Field(ge=0, description="Monthly income", examples=[5000])
    Coapplicant_Income: float = Field(ge=0, description="Monthly income", examples=[0])
    Age: int = Field(ge=18, le=100, examples=[35])
    Dependents: int = Field(ge=0, examples=[1])
    Credit_Score: float = Field(ge=300, le=900, examples=[700])
    Existing_Loans: int = Field(ge=0, examples=[1])
    DTI_Ratio: float = Field(ge=0, description="Debt-to-income ratio", examples=[0.3])
    Savings: float = Field(ge=0, examples=[10000])
    Collateral_Value: float = Field(ge=0, examples=[20000])
    Loan_Amount: float = Field(gt=0, examples=[15000])
    Loan_Term: float = Field(gt=0, description="Loan term, same unit as the training data", examples=[36])
    Employment_Status: str = Field(examples=["Salaried"])
    Marital_Status: str = Field(examples=["Married"])
    Loan_Purpose: str = Field(examples=["Personal"])
    Property_Area: str = Field(examples=["Urban"])
    Gender: str = Field(examples=["Male"])
    Employer_Category: str = Field(examples=["Private"])
    Education_Level: str = Field(examples=["Graduate"])


class Prediction(BaseModel):
    approved: bool
    approval_probability: float = Field(description="Probability the loan is approved, 0 to 1")
    threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.get("/options")
def options() -> dict:
    """Allowed values for every categorical field, taken from the training data."""
    return require_model().categories


@app.get("/model-info")
def model_info() -> dict:
    model = require_model()
    return {
        "model_name": model.model_name,
        "trained_at": model.trained_at,
        "sklearn_version": model.sklearn_version,
        "n_features": len(model.feature_columns),
        "metrics": model.metrics,
    }


@app.post("/predict", response_model=Prediction)
def predict(application: LoanApplication) -> dict:
    model = require_model()
    try:
        return model.predict(application.model_dump())
    except UnknownCategoryError as err:
        raise HTTPException(status_code=422, detail=str(err))
