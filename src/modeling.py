"""
modeling.py
-----------
Train/test splitting, scaling, model training, and evaluation for the
CreditWise loan approval model.

Change vs. the original notebook: `train_test_split` now uses
`stratify=y`. Loan_Approved is imbalanced (more "No" than "Yes"), and
without stratification the train/test split can end up with a
noticeably different approval rate in each split, making the
train/test metrics less comparable run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TARGET = "Loan_Approved"


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split features/target, then fit-transform a StandardScaler on train only."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@dataclass
class EvalResult:
    model_name: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion: list

    def __str__(self) -> str:
        return (
            f"{self.model_name}\n"
            f"  Precision : {self.precision:.4f}\n"
            f"  Recall    : {self.recall:.4f}\n"
            f"  F1 Score  : {self.f1:.4f}\n"
            f"  Accuracy  : {self.accuracy:.4f}\n"
            f"  Confusion Matrix : {self.confusion}"
        )


def evaluate(model_name: str, y_test, y_pred) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        accuracy=accuracy_score(y_test, y_pred),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
    )


def get_models() -> dict:
    """The three baseline classifiers used throughout the project."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }


def train_and_evaluate_all(X_train, X_test, y_train, y_test) -> list:
    """Fit every model in get_models() and return a list of EvalResult."""
    results = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append(evaluate(name, y_test, y_pred))
    return results
