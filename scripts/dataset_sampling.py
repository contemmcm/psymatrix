"""
Usage:

$ python  -m scripts.dataset_sampling --model "google-bert/bert-base-cased" --sample-size 500
"""

import os
import argparse

import numpy as np

from datasets import load_dataset
from psymatrix.finetune import finetune

SEED = 42
STEP_SIZE = 500
NUM_SAMPLES_PER_STEP = 250

parser = argparse.ArgumentParser(description="Dataset sampling experiments")

parser.add_argument(
    "--model",
    dest="model",
    type=str,
    required=True,
    help="The name of the pretrained model.",
)

parser.add_argument(
    "--sample-size",
    dest="sample_size",
    type=int,
    required=True,
    help="The sample size used for finetuning.",
)


def split_dataset(dataset, test_size=0.30):
    """
    Split the dataset into train, test, and validation sets.
    """
    dataset = dataset.shuffle(SEED).train_test_split(test_size=test_size)

    return dataset["train"], dataset["test"]


def sample_dataset(dataset, sample_size):
    """
    Sample a dataset.
    """

    return dataset.shuffle().select(range(sample_size))


def dump(original_dataset_id, original_split):
    dataset = load_dataset(original_dataset_id, split=original_split).shuffle(SEED)

    train_dataset, test_dataset = split_dataset(dataset)

    sample_sizes = np.arange(STEP_SIZE, train_dataset.num_rows, STEP_SIZE)

    for sample_size in sample_sizes:
        for sample_id in range(NUM_SAMPLES_PER_STEP):

            sampled_train_dataset = sample_dataset(train_dataset, sample_size)

            save_name = f"datasets/{original_dataset_id}_{original_split}_{sample_size:05d}_{sample_id:05d}"

            sampled_train_dataset.to_csv(f"{save_name}/train.csv")

            print(".", end="")

    save_name = f"datasets/{original_dataset_id}_{original_split}"

    test_dataset.to_csv(f"{save_name}/test.csv")


def eval(model_id, dataset_id):

    perf_file = (
        f"data/performance/datasets/contemmcm/{dataset_id}/test/{model_id}/metrics.json"
    )

    if os.path.exists(perf_file):
        print(f"Model {model_id} was already fine-tuned for dataset {dataset_id}.")
        return

    print("Fine-tuning model {} on dataset {}...".format(model_id, dataset_id))

    finetune(
        model_id,
        dataset_id,
        data_files={
            "train": "train.csv",
            "test": "../20_newsgroups_complete/test.csv",
        },
    )


def finetune_sampled_ds(model_id, sample_size):
    for sample_id in range(NUM_SAMPLES_PER_STEP):
        eval(
            model_id=model_id,
            dataset_id=f"datasets/contemmcm/20_newsgroups_complete_{sample_size:05d}_{sample_id:05d}",
        )


if __name__ == "__main__":
    # dump(original_dataset_id="contemmcm/20_newsgroups", original_split="complete")
    # eval(
    #     model_id="google-bert/bert-base-cased",
    #     dataset_id="datasets/contemmcm/20_newsgroups_complete_00500_00002",
    # )
    args = parser.parse_args()

    finetune_sampled_ds(model_id=args.model, sample_size=args.sample_size)
