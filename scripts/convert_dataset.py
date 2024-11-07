"""
This script converts a generic text dataset from Hugging Face into a format that is 
compatible with PsyMatrix.

Example usage:

```bash
$ python scripts/convert_dataset.py --name_or_path "SetFit/20_newsgroups" \
    --output_name "SetFit/20_newsgroups" \
    --train_split "train" \
    --test_split "test" \
    --content_cols "text" \
    --label_col "label"
```

To load the converted dataset in Python:

```python
from datasets import load_dataset

dataset = load_dataset("data/datasets/SetFit/20_newsgroups")
```

"""

import argparse
import os

import pandas as pd
from datasets import load_dataset

LOCAL_DATASET_BASEDIR = os.path.join("data", "datasets")
DEFAULT_CONTENT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"


parser = argparse.ArgumentParser(
    description="Converts Hugging Face datasets to PsyMatrix format."
)

parser.add_argument(
    "--name_or_path",
    dest="name_or_path",
    type=str,
    required=True,
    help="The name or path of the dataset.",
)

parser.add_argument(
    "--config_name",
    dest="config_name",
    type=str,
    required=False,
    default=None,
    help="The configuration name of the dataset.",
)

parser.add_argument(
    "--train_split",
    dest="train_split",
    type=str,
    required=True,
    help="The name of the train split.",
)

parser.add_argument(
    "--test_split",
    dest="test_split",
    type=str,
    required=True,
    help="The name of the test split.",
)

parser.add_argument(
    "--train_size",
    dest="train_size",
    type=float,
    required=False,
    help="The size of the train split.",
)

parser.add_argument(
    "--test_size",
    dest="test_size",
    type=float,
    required=False,
    help="The size of the test split.",
)

parser.add_argument(
    "--seed",
    dest="seed",
    type=int,
    required=False,
    help="The seed used to shuffle the dataset.",
)

parser.add_argument(
    "--content_cols",
    dest="content_cols",
    type=str,
    required=True,
)

parser.add_argument(
    "--label_col",
    dest="label_col",
    type=str,
    required=False,
)

parser.add_argument(
    "--output_name",
    dest="output_name",
    type=str,
    required=True,
)


def run():
    args = parser.parse_args()

    if args.train_split == args.test_split:
        train_test_ds = convert_dataset(
            args.name_or_path,
            args.config_name,
            split=args.train_split,
            train_size=_int_or_float(args.train_size),
            test_size=_int_or_float(args.test_size),
            seed=args.seed,
            content_cols=args.content_cols.split("+"),
        )
        train_ds = train_test_ds["train"]
        test_ds = train_test_ds["test"]
    else:
        train_ds = convert_dataset(
            args.name_or_path,
            args.config_name,
            split=args.train_split,
            seed=args.seed,
            content_cols=args.content_cols.split("+"),
        )
        test_ds = convert_dataset(
            args.name_or_path,
            args.config_name,
            split=args.test_split,
            seed=args.seed,
            content_cols=args.content_cols.split("+"),
        )

    save_dataset(train_ds, args.output_name, "train", args.label_col)
    save_dataset(test_ds, args.output_name, "test", args.label_col)


def convert_dataset(
    name_or_path,
    config_name,
    split,
    seed,
    content_cols,
    train_size=None,
    test_size=None,
):
    full_dataset = load_dataset(
        name_or_path,
        config_name,
        split=split,
        trust_remote_code=True,
    )

    if seed:
        full_dataset = full_dataset.shuffle(seed)

    if train_size or test_size:
        dataset = full_dataset.train_test_split(
            test_size=test_size, train_size=train_size, shuffle=False
        )
    else:
        dataset = full_dataset

    return combine_dataset_columns(dataset, content_cols, DEFAULT_CONTENT_COLUMN)


def combine_dataset_columns(dataset, columns, new_column_name, sep="\n\n"):
    """
    Combine the values of multiple columns into a new column.
    """

    if len(columns) == 1:
        if columns[0] == new_column_name:
            return dataset

        try:
            return dataset.rename_column(columns[0], new_column_name)
        except ValueError:
            # If there is already a column with the new name, remove it first
            return dataset.remove_columns(new_column_name).rename_column(
                columns[0], new_column_name
            )

    dataset = dataset.map(
        lambda example: {
            new_column_name: sep.join(
                [example[column] for column in columns if example[column]]
            )
        }
    )

    # Drop the original columns
    for column in columns:
        if column == new_column_name:
            continue
        dataset = dataset.remove_columns(column)

    return dataset


def save_dataset(ds, output_name, split, label_col):

    # keep only the columns "text" and "label"
    rows = [
        {
            DEFAULT_CONTENT_COLUMN: row[DEFAULT_CONTENT_COLUMN],
            DEFAULT_LABEL_COLUMN: row[label_col],
        }
        for row in ds
    ]

    df = pd.DataFrame(rows)

    output_file = os.path.join(LOCAL_DATASET_BASEDIR, output_name, split + ".csv")
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_file, index=False)


def _int_or_float(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


if __name__ == "__main__":
    run()
