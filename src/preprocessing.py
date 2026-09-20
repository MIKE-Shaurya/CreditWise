"""
preprocessing.py
-----------------
Data loading and cleaning utilities for the CreditWise loan approval model.

Responsibilities:
    * Load the raw CSV
    * Drop identifier columns that carry no predictive signal
    * Impute missing values (mean for numeric, mode for categorical)
    * Encode categorical columns (one-hot for nominal, label-encode for
      the ordinal/binary ones)

Bugs fixed vs. the original notebook:
    * `Applicant_ID` is now dropped *before* imputation/encoding instead of
      after, so it never wastes a slot in the numeric imputer.
    * The old code separated numeric/categorical columns once, at the top,
      and reused that list later even after columns had changed shape.
      Here each function recomputes what it needs, so it can't go stale.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ID_COLUMNS = ["Applicant_ID"]
TARGET = "Loan_Approved"

# Nominal categorical columns -> one-hot encoded (no natural order)
ONE_HOT_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

# Ordinal / binary columns -> label encoded
LABEL_ENCODE_COLS = ["Education_Level", TARGET]


def load_data(path: str) -> pd.DataFrame:
    """Read the raw loan application CSV."""
    return pd.read_csv(path)


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are unique identifiers and carry no signal."""
    cols_present = [c for c in ID_COLUMNS if c in df.columns]
    return df.drop(columns=cols_present)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with the column mean and missing
    categorical values with the column mode.
    """
    df = df.copy()

    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal categorical columns and label-encode the
    ordinal/binary columns (Education_Level, Loan_Approved).
    """
    df = df.copy()

    ohe_cols = [c for c in ONE_HOT_COLS if c in df.columns]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    for col in LABEL_ENCODE_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    return df


def clean_and_encode(path: str) -> pd.DataFrame:
    """Full pipeline: load -> drop IDs -> impute -> encode."""
    df = load_data(path)
    df = drop_id_columns(df)
    df = impute_missing(df)
    df = encode_features(df)
    return df
