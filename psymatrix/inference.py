import json

import pandas as pd
import torch
from pythae.models import VAE

from psymatrix.metafeatures.compression import vae
from psymatrix.nn2 import RankingModel
from psymatrix.utils import load_metafeatures


def run():

    metafeatures = load_metafeatures("emnlp24_datasets.txt", config="625", fillna=0)

    with open("metafeatures_stats.json", "r", encoding="utf8") as f:
        scaler = json.load(f)

    feature_names = list(metafeatures.columns)

    pd.options.mode.copy_on_write = True

    vae_model = VAE.load_from_folder(
        "data/vae/VAE_training_2024-08-19_10-07-25/final_model/"
    )

    net = RankingModel(input_size=2, hidden_size=256, num_classes=24)
    net.load("data/neural_net/fold_0_repeat_0_nnet.pth")
    net.eval()

    # Read model names
    with open("emnlp24_models.txt", encoding="utf8") as f:
        models = [line.strip() for line in f.readlines() if line.strip()]

    for i in range(146):

        data = metafeatures.iloc[i]  # Select the first dataset

        embeddings = vae.transform(vae_model, data)

        #
        # Calculate embeddings for the given dataset
        #

        # 1: Scale the input data between 0 and 1

        for name in feature_names:
            data[name] = (data[name] - scaler[name]["min"]) / (
                scaler[name]["max"] - scaler[name]["min"]
            )

        # 2: Load the VAE model and calculate the embeddings

        # print(embeddings)

        #
        # Estimate the performance of the models on the dataset
        #

        # 1. Load the trained predictor model

        # 2. Predict the performance of the models
        y_pred = net(torch.tensor(embeddings.to_numpy()))

        y_pred_rank = y_pred.argsort(dim=1, descending=True)

        idx = list(y_pred_rank.squeeze().numpy())

        for i in idx:
            print(models[i], y_pred[0, i])
            break


if __name__ == "__main__":
    run()
