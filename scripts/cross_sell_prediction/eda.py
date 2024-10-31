# eda.py

import pandas as pd


def perform_eda(df_train, df_test):
    """
    Perform EDA by combining training and test data for summary statistics.

    Parameters:
    - df_train: DataFrame of training data.
    - df_test: DataFrame of test data.

    Returns:
    - combined_df: DataFrame after combining train and test data.
    - numeric_summary: Summary of numerical features in combined data.
    - categorical_summary: Summary of categorical features in combined data.
    """
    # Add 'Response' column to test data with a placeholder value
    df_test['Response'] = -1

    # Concatenate train and test datasets
    combined_df = pd.concat([df_train, df_test], axis=0)
    print(f"Combined data shape: {combined_df.shape}")

    # Get summary statistics
    numeric_summary = combined_df.describe()
    categorical_summary = combined_df.describe(include='object')

    return combined_df, numeric_summary, categorical_summary
