import numpy as np
import pandas as pd


def drop_constant_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop constant features from the dataset.
    """
    constant_features = data.columns[data.nunique() == 1]
    input_features = np.array([w for w in data.columns if w not in constant_features])
    return data[input_features]


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.95):
    """
    Remove highly correlated columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The correlation threshold for removing columns. Default is 0.95.

    Returns:
    pd.DataFrame: DataFrame with highly correlated columns removed.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation greater than the threshold
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > threshold)
    ]

    # Drop the columns
    df_reduced = df.drop(columns=to_drop)

    return df_reduced
