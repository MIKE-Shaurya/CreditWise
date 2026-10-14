"""
train.py
--------
Train the CreditWise model and save everything the API needs to serve it
(model, scaler, feature order, category lists) as a single file.

    python -m src.train
    python -m src.train --data data/raw/loan_approval_data.csv --out models/creditwise_model.joblib
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import engineer_features
from src.modeling import TARGET, evaluate, split_and_scale
from src.preprocessing import ONE_HOT_COLS, clean_and_encode, load_data

DEFAULT_DATA = "data/raw/loan_approval_data.csv"
DEFAULT_OUT = "models/creditwise_model.joblib"
MODEL_NAME = "Logistic Regression"

# Categorical inputs the API has to accept (everything one-hot or label encoded).
CATEGORICAL_INPUTS = ONE_HOT_COLS + ["Education_Level"]


def train(data_path: str, out_path: str) -> dict:
    raw = load_data(data_path)

    # LabelEncoder sorts classes alphabetically, so "No" -> 0 and "Yes" -> 1.
    # The API reports probability of class 1, so make sure that holds.
    target_classes = sorted(raw[TARGET].dropna().unique())
    if target_classes != ["No", "Yes"]:
        raise ValueError(
            f"Expected {TARGET} to contain only 'Yes'/'No', found {target_classes}"
        )

    categories = {
        col: sorted(raw[col].dropna().unique().tolist()) for col in CATEGORICAL_INPUTS
    }
    # Same mapping LabelEncoder applied to Education_Level during training.
    education_map = {c: i for i, c in enumerate(categories["Education_Level"])}

    df = engineer_features(clean_and_encode(data_path))
    feature_columns = [c for c in df.columns if c != TARGET]

    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    result = evaluate(MODEL_NAME, y_test, model.predict(X_test))
    print(result)

    bundle = {
        "model_name": MODEL_NAME,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "categories": categories,
        "education_map": education_map,
        "metrics": {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "test_rows": int(len(y_test)),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"\nSaved model bundle to {out}")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the CreditWise model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to loan_approval_data.csv")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the model bundle")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
