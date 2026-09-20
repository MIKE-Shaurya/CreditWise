"""
feature_engineering.py
-----------------------
Engineered features for the CreditWise loan approval model.

What was wrong with the original feature-engineering cell
-----------------------------------------------------------
    df["DTI_Ratio_sq"]        = df["DTI_Ratio"] ** 2
    df["Credit_Score"]        = df["Credit_Score"] ** 2
    df["Applicant_Income_sq"] = np.log1p(df["Applicant_Income"])

    1. `Credit_Score` was squared *in place*, silently overwriting the
       original column. Squaring a score that's already on a roughly
       linear risk scale doesn't add information for a linear model and
       destroys interpretability (and reproducibility, since every cell
       run after it operates on the squared value).
    2. `Applicant_Income_sq` is actually a log transform, not a square -
       the name doesn't match what the code does.
    3. The new features weren't finance-domain features at all - just
       arbitrary polynomial transforms of columns that were already in
       the model. They didn't encode anything the raw columns didn't.
    4. The comparison at the end only re-trained KNN, so there was no
       way to tell whether the "engineered" features helped or hurt
       Logistic Regression / Naive Bayes too (in fact, KNN got *worse*:
       accuracy dropped from 0.76 -> 0.685, precisely because squaring
       Credit_Score distorts the distance metric KNN relies on).

What this module does instead
------------------------------
Adds features with real lending-domain meaning: affordability ratios,
collateral coverage, and log transforms of heavily right-skewed money
columns (income/savings/collateral/loan amount all have a wide range,
so a log transform tends to help linear models and KNN's distance
metric more than a raw square does).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6  # avoids divide-by-zero without materially changing ratios


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

    # How many years of total income the loan represents.
    df["Loan_to_Income_Ratio"] = df["Loan_Amount"] / (df["Total_Income"] + EPS)

    # Rough estimated monthly installment (loan spread over its term).
    df["Estimated_EMI"] = df["Loan_Amount"] / df["Loan_Term"].replace(0, np.nan)
    df["Estimated_EMI"] = df["Estimated_EMI"].fillna(df["Estimated_EMI"].mean())

    # What fraction of monthly income that installment would eat up.
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
