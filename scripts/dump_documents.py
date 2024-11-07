"""
This script dumps documents from Hugging Face datasets to the filesystem.

Example usage:

```bash
$ python scripts/dump_documents.py --name_or_path "SetFit/20_newsgroups" --split "train"
``` 
"""

import argparse
import os

from datasets import load_dataset

DOCUMENTS_BASEDIR = os.path.join("data", "documents")
DEFAULT_CONTENT_COLUMN = "text"

parser = argparse.ArgumentParser(
    description="Dump documents from Hugging Face datasets to filesystem."
)

parser.add_argument(
    "--name_or_path",
    dest="name_or_path",
    type=str,
    required=True,
    help="The name or path of the dataset.",
)

parser.add_argument(
    "--split",
    dest="split",
    type=str,
    required=True,
    help="The name of the split.",
)


def make_filename(name_or_path, split, i, n_digits, base_dir=""):
    return os.path.join(
        base_dir,
        name_or_path,
        split,
        "{}.txt".format(str(i).zfill(n_digits)),
    )


def run():
    args = parser.parse_args()

    for retries in range(3):
        try:
            dataset = load_dataset(args.name_or_path)
            break
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print(f"Retrying... ({retries + 1})")

    ds_split = dataset[args.split]

    n_digits = len(str(ds_split.num_rows))

    for i, doc in enumerate(ds_split[DEFAULT_CONTENT_COLUMN]):
        fname = make_filename(
            args.name_or_path, args.split, i, n_digits, DOCUMENTS_BASEDIR
        )

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        with open(fname, "w", encoding="utf8") as f:
            f.write(doc)

        print(".", end="", flush=True)


if __name__ == "__main__":
    run()
