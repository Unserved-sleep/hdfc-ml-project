"""
feature_engineering.py
======================
Domain-specific feature engineering functions for the HDFC loan dataset.
All functions accept and return a pandas DataFrame.

Individual Feature Constructors
--------------------------------
- add_total_income      : Sum of applicant and co-applicant income
- add_emi_ratio         : Ratio of existing EMIs to total income
- add_loan_income_ratio : Ratio of loan amount to total income
- add_monthly_savings   : Difference between total income and monthly expenses
- add_asset_loan_ratio  : Ratio of asset value to loan amount

Orchestrator
------------
- create_features : Apply all relevant feature constructors in the correct order;
                    skips optional features when the required source column is absent
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Individual Feature Constructors
# ---------------------------------------------------------------------------

def add_total_income(df):
    """
    Add 'Total_Income' = Applicant_Income + Coapplicant_Income.

    Combines both income sources into a single combined household income column.
    """
    df["Total_Income"] = (
        df["Applicant_Income"] +
        df["Coapplicant_Income"]
    )
    return df


def add_emi_ratio(df):
    """
    Add 'EMI_Income_Ratio' = Existing_EMIs / (Total_Income + 1).

    Measures the proportion of income already committed to EMI payments.
    A +1 offset prevents division by zero for zero-income records.

    Requires 'Total_Income' to already be present (call add_total_income first).
    """
    df["EMI_Income_Ratio"] = (
        df["Existing_EMIs"] /
        (df["Total_Income"] + 1)
    )
    return df


def add_loan_income_ratio(df):
    """
    Add 'Loan_Income_Ratio' = Loan_Amount / (Total_Income + 1).

    Indicates how many times the requested loan exceeds the applicant's income.
    A +1 offset prevents division by zero.

    Requires 'Total_Income' to already be present (call add_total_income first).
    """
    df["Loan_Income_Ratio"] = (
        df["Loan_Amount"] /
        (df["Total_Income"] + 1)
    )
    return df


def add_monthly_savings(df):
    """
    Add 'Monthly_Savings' = Total_Income - Monthly_Expense.

    Estimates discretionary income remaining after expenses.
    Only called when 'Monthly_Expense' exists in the DataFrame.
    """
    df["Monthly_Savings"] = (
        df["Total_Income"] -
        df["Monthly_Expense"]
    )
    return df


def add_asset_loan_ratio(df):
    """
    Add 'Asset_Loan_Ratio' = Asset_Value / (Loan_Amount + 1).

    Measures the collateral coverage relative to the loan size.
    Values > 1 indicate the asset covers the full loan amount.
    A +1 offset prevents division by zero for zero-loan records.
    """
    df["Asset_Loan_Ratio"] = (
        df["Asset_Value"] /
        (df["Loan_Amount"] + 1)
    )
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def create_features(df):
    """
    Apply all feature engineering steps in dependency order.

    Steps applied unconditionally:
      1. add_total_income      → required by downstream ratios
      2. add_emi_ratio
      3. add_loan_income_ratio

    Steps applied conditionally (column must exist in df):
      4. add_monthly_savings   → requires 'Monthly_Expense'
      5. add_asset_loan_ratio  → requires 'Asset_Value'

    Parameters
    ----------
    df : pandas DataFrame (raw or partially processed loan data)

    Returns
    -------
    df : same DataFrame with new engineered columns appended
    """
    # Core ratios — always added
    df = add_total_income(df)
    df = add_emi_ratio(df)
    df = add_loan_income_ratio(df)

    # Optional features — only added when source columns are available
    if "Monthly_Expense" in df.columns:
        df = add_monthly_savings(df)

    if "Asset_Value" in df.columns:
        df = add_asset_loan_ratio(df)

    return df
