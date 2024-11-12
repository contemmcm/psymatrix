import numpy as np

from datasets import load_dataset
from psymatrix.finetune import finetune

SEED = 42


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

    step_size = 500
    num_samples_per_step = 250

    sample_sizes = np.arange(step_size, train_dataset.num_rows, step_size)

    for sample_size in sample_sizes:
        for sample_id in range(num_samples_per_step):

            sampled_train_dataset = sample_dataset(train_dataset, sample_size)

            save_name = f"datasets/{original_dataset_id}_{original_split}_{sample_size:05d}_{sample_id:05d}"

            sampled_train_dataset.to_csv(f"{save_name}/train.csv")

            print(".", end="")

    save_name = f"datasets/{original_dataset_id}_{original_split}"

    test_dataset.to_csv(f"{save_name}/test.csv")


def eval(model_id, dataset_id):
    finetune(
        model_id,
        dataset_id,
        data_files={
            "train": "train.csv",
            "test": "../20_newsgroups_complete/test.csv",
        },
    )


if __name__ == "__main__":
    # dump(original_dataset_id="contemmcm/20_newsgroups", original_split="complete")
    eval(
        model_id="google-bert/bert-base-cased",
        dataset_id="datasets/contemmcm/20_newsgroups_complete_00500_00002",
    )
