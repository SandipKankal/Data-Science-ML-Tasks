import pandas as pd


def feature_engineering(df):
    # Example: Creating a new feature called 'Total Spend'
    df['Total Spend'] = df['Monthly Charges'] * df['Tenure']
    return df
