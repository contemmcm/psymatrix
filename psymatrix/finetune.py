"""
Usage:

$ python -m psymatrix.finetune \
    -m "google-bert/bert-base-cased" \
    -d "PsyMatrix/cls_agnews_SourceTitleTextVsLabel__BaseTop4Balanced"

$ python -m psymatrix.finetune \
    -e "emnlp24" \
    -d "PsyMatrix/cls_agnews_SourceTitleTextVsLabel__BaseTop4Balanced"

$ python -m psymatrix.finetune \
    -e "icml25" \
    -m "google-bert/bert-base-cased" \
    -d "PsyMatrix/cls_20newsgroups_SubjectTextVsLabel__BaseDefault"
"""

import argparse
from itertools import product
import json
import os
import time
from functools import partial
from shutil import rmtree
from typing import Union

import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset


MAX_TOKENS = 1024
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 3
SEED = 42

BATCH_SIZE = 8
LEARNING_RATE = 2.5e-6

DEFAULT_TRAINING_ARGS = {
    "optim": "adamw_torch",
    "overwrite_output_dir": True,
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": MAX_EPOCHS,
    "lr_scheduler_type": "linear",
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "logging_strategy": "steps",
    "logging_steps": 250,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "fp16": False,
}

parser = argparse.ArgumentParser(
    description="Finetune a pretrained model on a specific task-dataset."
)

parser.add_argument(
    "-m",
    "--model",
    dest="model_id",
    type=str,
    help="The pre-trained model ID. Overwrites model if used in conjunction with --experiment. E.g., 'google/bert-base-cased'.",
    required=False,
)

parser.add_argument(
    "-d",
    "--dataset",
    dest="dataset_id",
    type=str,
    help="The name or path of the dataset. Overwrites datasets if used in conjunction with --experiment. E.g., 'SetFit/20_newsgroups'.",
    required=False,
)

parser.add_argument(
    "-e",
    "--experiment",
    dest="experiment",
    help="The name of the experiment to load models and datasets from.",
    required=False,
)

parser.add_argument(
    "--train-split",
    dest="train_split",
    default="train",
    help="The name of the split to use for training.",
    required=False,
)

parser.add_argument(
    "--train-split-usage",
    dest="train_split_usage",
    default=100,
    type=int,
    help="The percentage of the train split to use for training (0-100).",
)

parser.add_argument(
    "--test-split",
    dest="test_split",
    default="test",
    help="The name of the split to use for testing.",
    required=False,
)

parser.add_argument(
    "--test-split-usage",
    dest="test_split_usage",
    default=100,
    type=int,
    help="The percentage of the test split to use for evaluation (0-100).",
)

parser.add_argument(
    "--no-cuda",
    dest="no_cuda",
    action="store_true",
    help="Disable CUDA.",
    required=False,
)


class SaveMetricsCallback(TrainerCallback):
    """
    Callback to save the metrics to
    """

    def __init__(
        self,
        fname_base: str,
        hyperparameters_id: int = None,
    ):
        self.metrics = []
        self.fname_base = fname_base
        self.is_disabled = False
        self.hyperparameters_id = hyperparameters_id
        super().__init__()

    def disable(self):
        self.is_disabled = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Save the metrics to a file.
        """
        if self.is_disabled:
            return

        # Adding timestamp to metrics
        metrics["timestamp"] = time.time()

        self.metrics.append(metrics)

        fname = f"{self.fname_base}/metrics_all.json"

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        with open(fname, "w", encoding="utf8") as f:
            json.dump(self.metrics, f, indent=2)


def get_num_labels(
    dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    label_column: str = "label",
    train_split: str = "train",
    test_split: str = "test",
):
    """
    Get the number of unique labels in the dataset.
    """
    labels = set(dataset[train_split][label_column])
    labels.update(set(dataset[test_split][label_column]))
    num_labels = len(labels)

    return num_labels


def get_tokenizer_max_length(model_id: str):
    """
    Get the maximum length of the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_max_length = tokenizer.model_max_length

    return min(tokenizer_max_length, MAX_TOKENS)


def get_tokenizer_args(model_id: str):
    """
    Return the best tokenizer arguments for the given model.
    """
    return {
        "max_length": get_tokenizer_max_length(model_id),
        "truncation": True,
        "padding": "max_length",
    }


def tokenize_function(tokenizer, model_id, examples):
    """
    Tokenize the examples for the given job.
    """
    tokenizer_args = get_tokenizer_args(model_id)

    # Tokenize the examples
    tokenized_inputs = tokenizer(examples["text"], **tokenizer_args)

    return tokenized_inputs


def compute_metrics_classification(eval_pred):
    """
    Compute the metrics for the classification task.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Top-1 accuracy
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def finetune(
    model_id: str,
    dataset_name_or_path: str,
    overwrite: bool = False,
    clean_up: bool = True,
    train_split: str = "train",
    test_split: str = "test",
    no_cuda: bool = False,
    hyperparameters_id: int = None,
    hyperparameters: dict = None,
    train_split_usage=100,
    test_split_usage=100,
):
    """
    Finetune the pre-trained model on the given dataset.
    """
    fname_model_base = os.path.join(
        dataset_name_or_path,
        model_id,
        f"train-{train_split_usage:03d}_test-{test_split_usage:03d}",
    )

    if hyperparameters_id is not None:
        fname_model_base = f"{fname_model_base}_hp-{hyperparameters_id:03d}"

    fname_base = os.path.join(
        "data",
        "performance",
        fname_model_base,
        test_split,
    )

    fname_output = os.path.join(fname_base, "metrics.json")

    if os.path.exists(fname_output) and not overwrite:
        with open(fname_output, "r", encoding="utf8") as f:
            metrics = json.load(f)
        return metrics

    dataset = load_dataset(dataset_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=get_num_labels(
            dataset, train_split=train_split, test_split=test_split
        ),
    )

    # Hack to bypass the contiguous error of some models
    for param in model.parameters():
        param.data = param.data.contiguous()

    # if DEVICE_COUNT > 1:
    #     model = torch.nn.DataParallel(model)

    _training_args = DEFAULT_TRAINING_ARGS.copy()

    if hyperparameters:
        if "learning_rate" in hyperparameters:
            _training_args["learning_rate"] = hyperparameters["learning_rate"]

        if "batch_size" in hyperparameters:
            _training_args["per_device_train_batch_size"] = hyperparameters[
                "batch_size"
            ]
            _training_args["per_device_eval_batch_size"] = hyperparameters["batch_size"]

    training_args = TrainingArguments(
        output_dir=f"data/finetune/{fname_model_base}",
        no_cuda=no_cuda,
        **_training_args,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        # model.resize_token_embeddings(len(tokenizer))

    tokenize = partial(
        tokenize_function,
        tokenizer,
        model_id,
    )

    if train_split_usage < 100:
        dataset[train_split] = (
            dataset[train_split]
            .shuffle(seed=SEED)
            .select(range(int(train_split_usage * len(dataset[train_split]) / 100)))
        )

    if test_split_usage < 100:
        dataset[test_split] = (
            dataset[test_split]
            .shuffle(seed=SEED)
            .select(range(int(test_split_usage * len(dataset[test_split]) / 100)))
        )

    train_dataset = dataset[train_split].map(tokenize, batched=True)
    test_dataset = dataset[test_split].map(tokenize, batched=True)

    save_callback = SaveMetricsCallback(fname_base, hyperparameters_id)

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        # data_collator=data_collator,  # Ensure proper padding
        callbacks=[
            save_callback,
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
        compute_metrics=compute_metrics_classification,
    )
    # Register initial timestamp
    metrics = trainer.evaluate()

    # Train the model
    trainer.train()

    # Evaluate the model (the best one is loaded automatically)
    save_callback.disable()
    metrics = trainer.evaluate()

    # Save metrics to a file
    os.makedirs(os.path.dirname(fname_output), exist_ok=True)

    with open(fname_output, "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)

    # Save training arguments
    with open(f"{fname_base}/training_args.json", "w", encoding="utf8") as f:
        json.dump(training_args.to_dict(), f, indent=2)

    if clean_up:
        # Clean up the training directory
        rmtree(training_args.output_dir)

    return metrics


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


def load_hyper_parameters(experiment: str):
    """
    Expand the hyperparameter optimization results.
    """
    fname = f"experiments/{experiment}/hpo.json"

    if not os.path.exists(fname):
        return None

    with open(fname, "r", encoding="utf8") as f:
        hpo = json.load(f)

    keys = list(hpo.keys())
    values = list(hpo.values())

    combinations = list(product(*values))

    return keys, combinations


def run():
    args = parser.parse_args()

    models = []
    datasets = []
    hpo = None

    if args.experiment:
        models = load_models(args.experiment)
        datasets = load_datasets(args.experiment)
        hpo = load_hyper_parameters(args.experiment)

    if args.model_id:
        models = [args.model_id]

    if args.dataset_id:
        datasets = [args.dataset_id]

    if not models or not datasets:
        raise ValueError("No models or datasets specified.")

    device = "cpu"
    device_count = 1

    if not args.no_cuda:
        if torch.cuda.is_available():
            device = "cuda"
            device_count = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            device = "mps"
            device_count = torch.mps.device_count()

    print(f"Device: {device} (x{device_count})")

    for model_id in models:
        for dataset_id in datasets:
            if hpo:  # Perform grid search over hyper parameters
                hpo_keys = hpo[0]
                hpo_values = hpo[1]

                for idx, value in enumerate(hpo_values):
                    hyperparameters = dict(zip(hpo_keys, value))
                    print(
                        f"Finetuning {model_id} on {dataset_id} with hyperparameters: {hyperparameters}"
                    )
                    metrics = finetune(
                        model_id,
                        dataset_id,
                        train_split=args.train_split,
                        test_split=args.test_split,
                        no_cuda=args.no_cuda,
                        hyperparameters_id=idx,
                        hyperparameters=hyperparameters,
                        train_split_usage=args.train_split_usage,
                        test_split_usage=args.test_split_usage,
                    )

            else:
                print(f"Finetuning {model_id} on {dataset_id}")
                metrics = finetune(
                    model_id,
                    dataset_id,
                    train_split=args.train_split,
                    test_split=args.test_split,
                    no_cuda=args.no_cuda,
                    train_split_usage=args.train_split_usage,
                    test_split_usage=args.test_split_usage,
                )
                print(metrics)


if __name__ == "__main__":
    run()
