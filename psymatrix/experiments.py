"""
This module contains functions for loading and saving experiment configurations.
"""

import os

import numpy as np
import pandas as pd

from psymatrix.utils import load_performance as load_performance_base
from psymatrix.utils import load_selected_metafeatures as load_metafeatures_base


def load_datasets(experiment):
    """
    Load a list of dataset IDs from the specified file.
    """
    fname = f"experiments/{experiment}/datasets.txt"

    with open(fname, "r", encoding="utf8") as f:
        datasets = [
            line.strip()
            for line in f.readlines()
            if not line.strip().startswith("#") and line.strip()
        ]

    return datasets


def load_models(experiment):
    """
    Load a list of model IDs from the specified file.
    """
    fname = f"experiments/{experiment}/models.txt"
    with open(fname, "r", encoding="utf8") as f:
        models = [line.strip() for line in f.readlines() if line.strip()]

    return models


def load_metafeatures(experiment: str, num_features: int):
    """
    Load the metafeatures from the specified file.
    """
    fname_datasets = f"experiments/{experiment}/datasets.txt"
    fname_selection = f"experiments/{experiment}/metafeatures.{num_features}.config"

    metafeatures = load_metafeatures_base(
        fname_datasets, fname_selection, normalize=False, fillna=0
    )

    return metafeatures


def dump_performance(experiment, metric):
    """
    Dump the performance values to a text file.
    """

    datasets = load_datasets(experiment)
    models = load_models(experiment)

    fname_base = os.path.join("experiments", experiment)
    fname_perf = os.path.join(fname_base, f"perf_{metric}.txt")
    fname_perf_norm = os.path.join(fname_base, f"perf_{metric}_norm.txt")

    with open(fname_perf, "w", encoding="utf8") as f:
        for m, model in enumerate(models):
            performances = load_performance_base(model, datasets, metric)

            for v, val in enumerate(performances[metric].to_numpy()):

                if metric == "loss":
                    val = -np.log(val)

                f.write(f"{val:0.5f}")

                if v < len(performances) - 1:
                    f.write(" ")

            if m < len(models) - 1:
                f.write("\n")

    df = pd.read_csv(fname_perf, sep=" ", header=None)

    # Normalize the performance values between 0 and 1, collumn-wise
    df = (df - df.min()) / (df.max() - df.min())

    df.to_csv(fname_perf_norm, sep=" ", header=False, index=False)

    return df


def load_performance(experiment, metric, normalized=False):
    """
    Load the performance values from a text file.
    """

    fname_base = os.path.join("experiments", experiment)
    fname_perf = os.path.join(fname_base, f"perf_{metric}.txt")
    fname_perf_norm = os.path.join(fname_base, f"perf_{metric}_norm.txt")

    if normalized:
        return pd.read_csv(fname_perf_norm, sep=" ", header=None)
    return pd.read_csv(fname_perf, sep=" ", header=None)


def load_embeddings(experiment: str, latent_dim: int):
    """
    Load the embeddings from the specified file.
    """
    fname = f"experiments/{experiment}/embeddings.{latent_dim}.txt"
    return pd.read_csv(fname, sep=" ", header=None)


if __name__ == "__main__":
    print(dump_performance("emnlp24", "accuracy"))
    print(dump_performance("emnlp24", "loss"))
