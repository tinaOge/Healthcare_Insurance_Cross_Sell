import pandas as pd


def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path, index_col='id')
    return df


def perform_eda(df):
    """Perform exploratory data analysis on the DataFrame."""
    shape = df.shape
    numeric_summary = df.describe()
    categorical_summary = df.describe(include='object')

    return shape, numeric_summary, categorical_summary
