"""
Module for ranking pre-trained LLMs using PsyMatrix dataset embeddings and performance
data from previous experiments (meta-learning).
"""

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pythae.models import VAE
from sklearn.preprocessing import MinMaxScaler

from psymatrix.metafeatures.compression import vae
from psymatrix.metafeatures.compression.pca import perform_pca as train_pca
from psymatrix.metafeatures.selection.k_means import dimensionality_reduction as k_means
from psymatrix.nn2 import train_nnet as train_predictor
from psymatrix.utils import load_metafeatures


def eval_net(
    data: DataFrame,
    target: DataFrame,
    features: list = (),
    num_features: int = None,
    embedding_size: int = None,
    embedding_method: str = "pca",
    num_repeats: int = 5,
):
    if features and num_features:
        raise ValueError("Only one of features or num_features can be provided.")

    #
    # Feature seleciton
    #
    if num_features:
        features = k_means(data, num_clusters=num_features)

    if features:
        data = data[features]

    #
    # Embedding
    #
    scaler = MinMaxScaler(feature_range=(0, 1))

    if embedding_size:
        if embedding_method.startswith("vae"):

            data.to_csv("data.csv", index=False)
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
            data.to_csv("data_scaled.csv", index=False)

            if embedding_method == "vae":
                data = vae.fit_transform(
                    data, latent_dim=embedding_size, output_dir="data/vae/"
                )
            elif embedding_method == "vae-file":
                model = VAE.load_from_folder(
                    "data/vae/VAE_training_2024-08-19_10-07-25/final_model"
                )
                data = vae.transform(model, data)

            else:
                raise ValueError(f"Unknown VAE embedding method: {embedding_method}")

            data.to_csv("data_embeddings.csv", index=False)

        elif embedding_method == "pca":
            data, _ = train_pca(data, embedding_size, save=False)
        elif embedding_method == "file":
            data = pd.read_csv(
                f"vae_embeddings/{embedding_size:03d}_embeddings.csv",
            )
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")

    #
    # Train and eval model prediction network
    #

    # Normalize the data (again), as the embeddings might have changed the scale
    # data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # data.to_csv("data_scaled.csv2", index=False)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(data.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(target.to_numpy(), dtype=torch.float32).transpose(0, 1)

    psy_auc, psy_topk_vec_mean, psy_topk_vec_std = train_predictor(
        X_tensor, y_tensor, k_folds=10, num_repeats=num_repeats
    )

    return psy_auc, psy_topk_vec_mean, psy_topk_vec_std


def load_targets():
    """
    Load the performance data.
    """
    df = pd.read_csv("performance.txt", sep=" ", header=None)
    return df


def auc_single_model_policy():
    """
    Compute the AUC for a single model policy.
    """
    ranking = pd.read_csv("ranking.txt", sep=" ", header=None)

    num_models, num_datasets = ranking.shape

    topk_mat = np.zeros((num_models, num_models))
    auc_vec = np.zeros(num_models)

    # Read model names
    with open("emnlp24_models.txt", encoding="utf8") as f:
        models = [line.strip() for line in f.readlines() if line.strip()]

    for i in range(num_models):
        for k in range(num_models):
            topk_mat[i, k] = np.sum(ranking.iloc[i] <= (k + 1)) / num_datasets

    # Calculate the AUC
    for i in range(num_models):
        success_rate = topk_mat[i]
        auc_vec[i] = np.trapz(success_rate, axis=0) / num_models

    # Print in latex table format as follows: AUC & Top-1 & Top-3 & Top-5 & Top-10
    for i in range(num_models):
        print(
            f"{models[i]} & {auc_vec[i]:0.3f} & {topk_mat[i, 0]:0.3f} & {topk_mat[i, 2]:0.3f} & {topk_mat[i, 4]:0.3f} & {topk_mat[i, 9]:0.3f}"
        )


if __name__ == "__main__":
    metafeatures = load_metafeatures("emnlp24_datasets.txt", config="625", fillna=0)
    targets = load_targets()

    auc_single_model_policy()

    auc, topk_mean_vec, topk_std_vec = eval_net(
        metafeatures,
        targets,
        num_features=None,  # Skip feature selection
        embedding_size=2,
        embedding_method="vae",
        num_repeats=5,
    )

    print(f"AUC: {auc}")
    print(f"Top-k mean: {topk_mean_vec}")
    print(f"Top-k std: {topk_std_vec}")


# AUC: 0.8493313492063493
# Top-k mean: [0.3247619  0.49457143 0.52080952 0.60495238 0.68109524 0.79509524
#  0.9077619  0.91833333 0.94557143 0.968      0.98285714 0.98685714
#  0.98685714 0.98685714 0.989      0.99042857 0.99042857 0.99247619
#  0.99457143 0.99457143 0.9952381  0.99728571 0.99795238 1.        ]
# Top-k std: [0.13716116 0.1533745  0.14890383 0.13318222 0.12451536 0.10783118
#  0.07297255 0.05925172 0.04716153 0.04573276 0.02947081 0.02701847
#  0.02701847 0.02701847 0.0247266  0.02319868 0.02319868 0.01843473
#  0.01490681 0.01490681 0.01290681 0.00814286 0.00614286 0.        ]
