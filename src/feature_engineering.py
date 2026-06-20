import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Creates hour-based features from the elapsed 'Time' column.
    """
    df_engineered = df.copy()
    if 'Time' in df_engineered.columns:
        df_engineered['Time_Hr'] = df_engineered['Time'] / 3600.0
    return df_engineered

def create_amount_categories(df):
    """
    Categorizes transaction amounts into economic stratification buckets.
    """
    df_engineered = df.copy()
    if 'Amount' in df_engineered.columns:
        bins = [0, 50, 200, 1000, np.inf]
        labels = ['Micro (0-50)', 'Small (50-200)', 'Medium (200-1000)', 'Large (1000+)']
        df_engineered['Amount_Category'] = pd.cut(
            df_engineered['Amount'], bins=bins, labels=labels, right=False
        )
    return df_engineered

def apply_feature_engineering(df):
    """
    Applies all feature engineering transformations.
    """
    df = create_time_features(df)
    df = create_amount_categories(df)
    return df
