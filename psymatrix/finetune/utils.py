import re

from typing import Union

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset

from transformers import AutoTokenizer

from .constants import DEFAULT_MAX_TOKENS


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


def get_tokenizer_max_length(model_id: str, max_tokens: int):
    """
    Get the maximum length of the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_max_length = tokenizer.model_max_length

    return min(tokenizer_max_length, max_tokens)


def get_tokenizer_args(model_id: str, max_tokens: int):
    """
    Return the best tokenizer arguments for the given model.
    """
    return {
        "max_length": get_tokenizer_max_length(model_id, max_tokens),
        "truncation": True,
        "padding": "max_length",
    }


def tokenize_function(
    tokenizer, model_id, hyperparameters, examples, input_col="text", input_format=None
):
    """
    Tokenize the examples for the given job.
    """
    if hyperparameters and "max_tokens" in hyperparameters:
        max_tokens = hyperparameters["max_tokens"]
    else:
        max_tokens = DEFAULT_MAX_TOKENS

    tokenizer_args = get_tokenizer_args(model_id, max_tokens=max_tokens)

    if input_format:
        merged_examples = []
        cols = re.findall(r"{(.*?)}", input_format)
        num_examples = len(examples[cols[0]])

        for example_id in range(num_examples):
            values = {col: examples[col][example_id] for col in cols}
            example = input_format.format(**values)
            example.replace("\\n", "\n")
            example.replace("\\t", "\t")

            merged_examples.append(example)

        _examples = merged_examples
    else:
        _examples = examples[input_col]

    # Tokenize the examples
    tokenized_inputs = tokenizer(_examples, **tokenizer_args)

    return tokenized_inputs
