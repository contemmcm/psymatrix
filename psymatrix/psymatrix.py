"""
Usage:

$ python -m psymatrix.psymatrix \
    --experiment emnlp24 \
    --feature-selection 625 \
    --feature-compression 2 \
    --hidden-size=32 \
    --train-ranking
"""

import argparse
import configparser
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from pythae.models import VAE
from sklearn.preprocessing import MinMaxScaler

from psymatrix import experiments
from psymatrix.metafeatures.compression import vae
from psymatrix.metafeatures.selection import k_means, pre_processing
from psymatrix.nn2 import RankingModel
from psymatrix.nn2 import train_nnet as train_predictor
from psymatrix.scaler import Scaler
from psymatrix.utils import load_metafeatures, load_selected_metafeatures


def build_selected_metafeatures_file_name(experiment: str, num_features: int):
    return os.path.join(
        "experiments", experiment, f"metafeatures.{num_features}.config"
    )


def build_embeddings_file_name(experiment: str, latent_dim: int):
    return os.path.join("experiments", experiment, f"embeddings.{latent_dim}.txt")


class PsyMatrix:
    """
    Class to predict the ranking of the models for a given dataset.
    """

    def __init__(self, experiment: str, hidden_size: int = 32):
        self.experiment = experiment

        self.config = configparser.ConfigParser()
        self.config.read(os.path.join("experiments", experiment, "config.ini"))

        self.scaler = Scaler(experiment)
        self._metafeatures = None

        self.hidden_size = hidden_size
        self.num_classes = 24

        # Read model names
        self.llms = experiments.load_models(experiment)

        self.vae_model = VAE.load_from_folder(self.config["VAE"]["ModelPath"])

        nn_models = glob(os.path.join(self.config["NN"]["ModelBasePath"], "*_nnet.pth"))
        self.net = [None] * len(nn_models)

        for i, nn_path in enumerate(nn_models):
            self.net[i] = RankingModel(
                input_size=2, hidden_size=self.hidden_size, num_classes=self.num_classes
            )
            self.net[i].load(nn_path)

    def predict_rank(self, dataset_id: str, split: str = "train", verbose: bool = True):
        """
        Predict the ranking of the models for the given dataset.
        """
        embeddings = self.get_embeddings(dataset_id, split=split)
        pred_mean, _ = self.predict(embeddings)

        if verbose:
            ranking = pred_mean.sort_values()[::-1]
            for i, model in enumerate(ranking.index):
                print(f"{i+1}.\t{ranking[model]:.4f}\t{model}")

        return pred_mean

    def get_embeddings(self, dataset_id: str, split: str):
        """
        Get the embeddings for the given dataset.
        """
        selection = os.path.join(
            "experiments", self.experiment, "metafeatures.625.config"
        )
        metafeatures = load_selected_metafeatures(
            [dataset_id], selection, split=split, fillna=0.0
        ).iloc[0]

        return self.embed(metafeatures)

    def embed(self, dataset_metafeatures: pd.Series):
        """
        Embed the given dataset metafeatures.
        """
        data = self.scaler.transform(dataset_metafeatures)
        embeddings = vae.transform(self.vae_model, data)

        return torch.tensor(embeddings.to_numpy(), dtype=torch.float32)

    def predict(self, embeddings: torch.Tensor):
        """
        Predict the ranking of the models for the given dataset.
        """

        rankings = pd.DataFrame(
            np.zeros((len(self.net), len(self.llms))), columns=self.llms
        )

        for i, net in enumerate(self.net):
            rankings.iloc[i] = net(embeddings).detach().numpy().squeeze()

        return rankings.mean(), rankings.std()

    def get_pretrained_models(self):
        """
        Get the list of pretrained models.
        """
        return self.llms

    def metafeature_selection(
        self, num_features: int, method="kmeans", verbose: bool = True
    ):
        fout = build_selected_metafeatures_file_name(self.experiment, num_features)

        metafeatures = self._load_metafeatures()
        n_features = metafeatures.shape[1]

        if verbose:
            print(f"Initial number of metafeatures: {n_features}")

        metafeatures = pre_processing.drop_constant_features(metafeatures)
        n_features_after1 = metafeatures.shape[1]

        if verbose:
            print("After dropping constant features:", n_features_after1)

        threshold = 0.95
        metafeatures = pre_processing.drop_correlated_features(
            metafeatures, threshold=threshold
        )
        n_features_after2 = metafeatures.shape[1]

        if verbose:
            print(
                f"After dropping correlated features (treshold={threshold}):",
                n_features_after2,
            )

        if method == "kmeans":
            k_means.dimensionality_reduction(
                metafeatures,
                num_clusters=num_features,
                fout=fout,
            )
            if verbose:
                print(f"Metafeatures selected ({method}): {num_features}")
        else:
            raise ValueError(f"Unknown method: {method}")

    def train_metafeature_compressor(
        self, num_features: int, latent_dim: int, method="vae"
    ):
        """
        Train a metafeature compressor.
        """
        ds_path = os.path.join("experiments", self.experiment, "datasets.txt")
        sl_path = build_selected_metafeatures_file_name(self.experiment, num_features)
        data = load_selected_metafeatures(ds_path, sl_path, fillna=0.0)

        if method == "vae":

            scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize to [0, 1]

            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            self.vae_model = vae.fit(
                data, latent_dim=latent_dim, output_dir="data/vae/"
            )

            data = vae.transform(self.vae_model, data)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save the embeddings of each dataset to a file (for debugging purposes)
        fname_ds_embeddings = build_embeddings_file_name(self.experiment, latent_dim)

        data.to_csv(fname_ds_embeddings, index=False, sep=" ", header=False)

    def train_ranking_model(
        self,
        num_features: int = None,
        num_latent_features: int = None,
        metric: str = "accuracy",
        k_folds: int = 10,
        num_repeats: int = 5,
        hidden_size: int = 32,
    ):
        """
        Train the ranking model.
        """
        if num_latent_features:
            inputs = experiments.load_embeddings(self.experiment, num_latent_features)
        elif num_features:
            inputs = experiments.load_metafeatures(self.experiment, num_features)
        else:
            raise ValueError("Either num_features or num_latent_features must be set.")

        targets = experiments.load_performance(self.experiment, metric, normalized=True)

        X_tensor = torch.tensor(inputs.to_numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(targets.to_numpy(), dtype=torch.float32).transpose(0, 1)

        (
            y_pred_rank_all,
            y_real_rank_all,
            psy_auc,
            psy_topk_vec_mean,
            psy_topk_vec_std,
        ) = train_predictor(
            X_tensor,
            y_tensor,
            k_folds=k_folds,
            num_repeats=num_repeats,
            hidden_size=hidden_size,
        )

        pd.DataFrame(y_pred_rank_all).to_csv(
            f"experiments/{self.experiment}/predictions.txt", index=False, sep=" "
        )
        pd.DataFrame(y_real_rank_all).to_csv(
            f"experiments/{self.experiment}/correct.txt", index=False, sep=" "
        )

        return psy_auc, psy_topk_vec_mean, psy_topk_vec_std

    def _load_metafeatures(self):
        if self._metafeatures:
            return self._metafeatures

        ds_path = os.path.join("experiments", self.experiment, "datasets.txt")

        self._metafeatures = load_metafeatures(
            ds_path, include_datasets_ids=False, fillna=0.0
        )

        return self._metafeatures


def run():
    experiment = "emnlp24"

    pm = PsyMatrix(experiment)

    # embeddings = pm.embed(dataset)
    # embeddings = torch.tensor([[0.3865, -0.4295]], dtype=torch.float32)
    # print("embeddings1", embeddings)
    # predictions_mu, std = pm.predict(embeddings)

    n_steps = 100
    mu_1 = np.linspace(-2, 1.5, n_steps)
    mu_2 = np.linspace(-2, 4, n_steps)

    rank_mu = {}
    rank_std = {}

    for model in pm.get_pretrained_models():
        rank_mu[model] = np.zeros((n_steps, n_steps))
        rank_std[model] = np.zeros((n_steps, n_steps))

    for i in range(n_steps):
        for j in range(n_steps):
            dataset = torch.tensor([[mu_1[i], mu_2[j]]], dtype=torch.float32)
            predictions_mu, predictions_std = pm.predict(dataset)

            for m in predictions_mu.index:
                rank_mu[m][i, j] = predictions_mu[m]
                rank_std[m][i, j] = predictions_std[m]

    # save rank to file as csv
    for model in pm.get_pretrained_models():
        fname_mu = f"results/{model}_mu.txt"
        fname_std = f"results/{model}_std.txt"
        os.makedirs(os.path.dirname(fname_mu), exist_ok=True)

        pd.DataFrame(rank_mu[model]).to_csv(
            fname_mu, index=False, sep=" ", header=False
        )
        pd.DataFrame(rank_std[model]).to_csv(
            fname_std, index=False, sep=" ", header=False
        )


def main():
    parser = argparse.ArgumentParser(description="PsyMatrix")
    parser.add_argument(
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--feature-selection",
        dest="num_features",
        type=int,
        required=False,
        help="Number of features to select.",
    )
    parser.add_argument(
        "--feature-compression",
        dest="latent_dim",
        type=int,
        required=False,
        help="Latent dimension of the feature compressor.",
    )
    parser.add_argument(
        "--train-ranking",
        dest="train_ranking",
        action="store_true",
        required=False,
        help="Train the ranking model.",
    )
    parser.add_argument(
        "--target-dataset",
        dest="target_dataset",
    )
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=32)
    parser.add_argument("--train-split", dest="train_split", type=str, default="train")

    args = parser.parse_args()

    pm = PsyMatrix(args.experiment, args.hidden_size)

    if args.target_dataset:
        pm.predict_rank(args.target_dataset, args.train_split, verbose=True)
        return

    if args.num_features:
        cache_file = build_selected_metafeatures_file_name(
            args.experiment, args.num_features
        )
        if not os.path.exists(cache_file):
            pm.metafeature_selection(args.num_features)

    if args.latent_dim:
        cache_file = build_embeddings_file_name(args.experiment, args.latent_dim)
        if not os.path.exists(cache_file):
            pm.train_metafeature_compressor(args.num_features, args.latent_dim)

    if args.train_ranking:
        pm.train_ranking_model(
            num_features=args.num_features,
            num_latent_features=args.latent_dim,
            metric="loss",
            num_repeats=1,
        )


if __name__ == "__main__":
    main()
