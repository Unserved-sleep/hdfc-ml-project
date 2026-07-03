import pandas as pd
import numpy as np

def add_total_income(df):
    df["Total_Income"] = (
        df["Applicant_Income"] +
        df["Coapplicant_Income"]
    )
    return df

def add_emi_ratio(df):
    df["EMI_Income_Ratio"] = (
        df["Existing_EMIs"] /
        (df["Total_Income"] + 1)
    )
    return df

def add_loan_income_ratio(df):
    df["Loan_Income_Ratio"] = (
        df["Loan_Amount"] /
        (df["Total_Income"] + 1)
    )
    return df

def add_monthly_savings(df):
    df["Monthly_Savings"] = (
        df["Total_Income"] -
        df["Monthly_Expense"]
    )
    return df

def add_asset_loan_ratio(df):
    df["Asset_Loan_Ratio"] = (
        df["Asset_Value"] /
        (df["Loan_Amount"] + 1)
    )
    return df

def create_features(df):
    df = add_total_income(df)
    df = add_emi_ratio(df)
    df = add_loan_income_ratio(df)

    if (
        "Monthly_Expense" in df.columns
    ):
        df = add_monthly_savings(df)

    if (
        "Asset_Value" in df.columns
    ):
        df = add_asset_loan_ratio(df)

    return df