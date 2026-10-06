"""
feature_engineering.py
-----------------------
Engineered features for the CreditWise loan approval model.

-----------------------------------------------------------
    df["DTI_Ratio_sq"]        = df["DTI_Ratio"] ** 2
    df["Credit_Score"]        = df["Credit_Score"] ** 2
    df["Applicant_Income_sq"] = np.log1p(df["Applicant_Income_sq"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def add_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine applicant + co-applicant income and log-transform it."""
    df = df.copy()
    df["Total_Income"] = df["Applicant_Income"] + df["Coapplicant_Income"]
    df["Log_Total_Income"] = np.log1p(df["Total_Income"])
    return df


def add_affordability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios that describe how big the loan is relative to what the
    applicant can actually pay, which is what a lender cares about far
    more than any single raw column on its own.
    """
    df = df.copy()
    df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / (df["Total_Income"] + EPS)   .
    df["Estimated_EMI"] = df["Loan_Amount"] / df["Loan_Term"].replace(0, np.nan)
    df["Estimated_EMI"] = df["Estimated_EMI"].fillna(df["Estimated_EMI"].mean())

   
    monthly_income = df["Total_Income"] / 12
    df["EMI_to_Income_Ratio"] = df["Estimated_EMI"] / (monthly_income + EPS)

    return df


def add_cushion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How well protected the lender/applicant is if repayment gets tight:
    savings on hand and collateral value, both relative to loan size.
    """
    df = df.copy()
    df["Savings_to_Loan_Ratio"] = df["Savings"] / (df["Loan_Amount"] + EPS)
    df["Collateral_Coverage_Ratio"] = df["Collateral_Value"] / (df["Loan_Amount"] + EPS)
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform the money columns that are heavily right-skewed
    (wide min-max range relative to the median). This helps
    scale-sensitive/distance-based models (KNN, Logistic Regression)
    without discarding the original columns, unlike the original
    notebook which overwrote Credit_Score in place.
    """
    df = df.copy()
    for col in ["Savings", "Collateral_Value", "Loan_Amount"]:
        df[f"Log_{col}"] = np.log1p(df[col])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-engineering pipeline. Applied AFTER preprocessing
    (missing values imputed, categoricals encoded) but works on the
    still-numeric Total_Income/Loan_Amount/etc. columns, so run it
    before scaling.

    Unlike the original cell, this never drops or overwrites a raw
    column in place - engineered features are added alongside the
    originals, so downstream code (and you, reading the dataframe
    later) can always see both.
    """
    df = add_income_features(df)
    df = add_affordability_features(df)
    df = add_cushion_features(df)
    df = add_log_transforms(df)
    return df
