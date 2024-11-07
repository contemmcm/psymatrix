import pandas as pd
import torch
from pythae.data.preprocessors import DataProcessor
from pythae.models import VAE, VAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from sklearn.model_selection import train_test_split

from psymatrix.utils import load_metafeatures


def fit(
    data: pd.DataFrame,
    latent_dim: int = 2,
    batch_size: int = 32,
    output_dir: str = None,
    load_from_folder: str = None,
):
    """
    Fit a VAE model to the data.
    """
    if not (data.shape[1] ** 0.5).is_integer():
        raise ValueError("The number of features in the data must be a perfect square.")

    trainer_config = BaseTrainerConfig(
        num_epochs=200,
        learning_rate=1e-3,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    dim = int(data.shape[1] ** 0.5)

    vae_config = VAEConfig(
        input_dim=(1, dim, dim),
        latent_dim=latent_dim,
    )

    processor = DataProcessor()
    train_data = processor.process_data(data.to_numpy())
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Hack to transform the metafeatures into a image-like tensor
    train_data = torch.stack([row.reshape(dim, dim) for row in train_data])
    test_data = torch.stack([row.reshape(dim, dim) for row in test_data])

    if load_from_folder:
        model = VAE.load_from_folder(load_from_folder)
    else:
        model = VAE(model_config=vae_config)
        pipeline = TrainingPipeline(training_config=trainer_config, model=model)
        pipeline(train_data=train_data, eval_data=test_data)

    return model


def transform(model: VAE, data: pd.DataFrame, include_covariance: bool = False):
    processor = DataProcessor()

    data = processor.process_data(data.to_numpy())

    if len(data.shape) == 1:
        dim = int(data.shape[0] ** 0.5)
        data = data.reshape(dim, dim)
    else:
        dim = int(data.shape[1] ** 0.5)
        # Hack to transform the metafeatures into a image-like tensor
        data = torch.stack([row.reshape(dim, dim) for row in data])

    model = model.to("cpu")
    model_output = model.encoder(data)

    df_mu = pd.DataFrame(
        model_output["embedding"].detach().numpy(),
        columns=[f"mu_{i}" for i in range(model.latent_dim)],
    )

    if include_covariance:
        df_logcov = pd.DataFrame(
            model_output["log_covariance"].detach().numpy(),
            columns=[f"logcov_{i}" for i in range(model.latent_dim)],
        )

        df_concat = pd.concat([df_mu, df_logcov], axis=1)

        return df_concat

    return df_mu


def fit_transform(
    data: pd.DataFrame,
    latent_dim: int = 2,
    batch_size: int = 32,
    output_dir: str = None,
    load_from_folder: str = None,
):
    model = fit(
        data,
        batch_size=batch_size,
        latent_dim=latent_dim,
        output_dir=output_dir,
        load_from_folder=load_from_folder,
    )

    # Move model to cpu
    if model.device != "cpu":
        model.to("cpu")

    return transform(model, data)


def run(latent_dim: int = 4):
    data = load_metafeatures(
        "emnlp24_datasets.txt", config="base", fillna=0, normalize=True
    )

    data_transformed = fit_transform(data, latent_dim=latent_dim)

    # Save the embeddings to file
    df_mu = pd.DataFrame(
        data_transformed["embedding"].detach().numpy(),
        columns=[f"mu_{i}" for i in range(latent_dim)],
    )
    df_logcov = pd.DataFrame(
        data_transformed["log_covariance"].detach().numpy(),
        columns=[f"logcov_{i}" for i in range(latent_dim)],
    )
    df_concat = pd.concat([df_mu, df_logcov], axis=1)

    df_mu.to_csv(
        f"vae_embeddings/{latent_dim:03d}_embeddings_mu.txt", index=False, sep=" "
    )
    df_logcov.to_csv(
        f"vae_embeddings/{latent_dim:03d}_embeddings_logcov.txt", index=False, sep=" "
    )
    df_concat.to_csv(f"vae_embeddings/{latent_dim:03d}_embeddings.csv", index=False)


if __name__ == "__main__":
    run()
