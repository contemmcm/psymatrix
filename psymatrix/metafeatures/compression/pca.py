import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from psymatrix.utils import load_metafeatures


def perform_pca(df, n_components, save=True):
    """
    Perform PCA on a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n_components (int): Number of principal components to keep.

    Returns:
    pd.DataFrame: DataFrame with the principal components.
    PCA: The fitted PCA object.
    """
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)

    # Create a DataFrame with the principal components
    columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)

    if save:
        df_pca.to_csv(
            f"pca_embeddings/{n_components:03d}_embeddings.txt",
            index=False,
            header=False,
            sep=" ",
        )
        df_pca.to_csv(
            f"pca_embeddings/{n_components:03d}_embeddings.csv",
            index=False,
        )

    return df_pca, pca


def apply_pca(n_components):
    df = load_metafeatures("emnlp24_datasets.txt", normalize=True, fillna=0)

    return perform_pca(df, n_components)


def run():
    for n_components in [2, 4, 8, 16, 32, 64]:
        apply_pca(n_components)


if __name__ == "__main__":
    run()
