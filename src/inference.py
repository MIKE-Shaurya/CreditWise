"""
inference.py
------------
Turn one raw applicant record into the exact feature vector the model was
trained on, and score it.

The training pipeline (preprocessing.py) fits its encoders on the whole CSV
and throws them away, so they can't be reused for a single new applicant.
Instead, train.py saves the category lists and the final feature order, and
this module rebuilds the same columns from those.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.feature_engineering import engineer_features
from src.preprocessing import ONE_HOT_COLS

NUMERIC_INPUTS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]
CATEGORICAL_INPUTS = ONE_HOT_COLS + ["Education_Level"]


class UnknownCategoryError(ValueError):
    """A categorical value that the model never saw during training."""

    def __init__(self, column: str, value: str, allowed: list[str]):
        self.column = column
        self.value = value
        self.allowed = allowed
        super().__init__(f"{column}={value!r} is not one of {allowed}")


@dataclass
class CreditWiseModel:
    model_name: str
    model: object
    scaler: object
    feature_columns: list
    categories: dict
    education_map: dict
    metrics: dict
    trained_at: str
    sklearn_version: str

    @classmethod
    def load(cls, path: str | Path) -> "CreditWiseModel":
        bundle = joblib.load(path)
        if bundle["sklearn_version"] != sklearn.__version__:
            warnings.warn(
                f"Model was trained with scikit-learn {bundle['sklearn_version']} "
                f"but this environment has {sklearn.__version__}. Retrain in the "
                "same environment (e.g. `docker compose run --rm train`) if "
                "predictions look off.",
                stacklevel=2,
            )
        return cls(**bundle)

    def validate_categories(self, record: dict) -> None:
        for col in CATEGORICAL_INPUTS:
            if record[col] not in self.categories[col]:
                raise UnknownCategoryError(col, record[col], self.categories[col])

    def build_features(self, record: dict) -> pd.DataFrame:
        """One applicant -> one row with exactly the training feature columns."""
        row = pd.DataFrame([record])
        row["Education_Level"] = row["Education_Level"].map(self.education_map)
        row = pd.get_dummies(row, columns=ONE_HOT_COLS, dtype=float)
        row = engineer_features(row)
        # Reindexing zero-fills the one-hot columns for categories this applicant
        # doesn't have (including the one dropped by drop="first" at training).
        return row.reindex(columns=self.feature_columns, fill_value=0.0).astype(float)

    def predict(self, record: dict, threshold: float = 0.5) -> dict:
        self.validate_categories(record)
        features = self.build_features(record)
        scaled = self.scaler.transform(features.to_numpy())
        probability = float(self.model.predict_proba(scaled)[0, 1])
        return {
            "approved": probability >= threshold,
            "approval_probability": probability,
            "threshold": threshold,
        }
