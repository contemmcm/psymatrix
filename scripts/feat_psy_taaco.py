"""
This script runs TAACO on all datasets in the database. It requires the dataset to
be exported using the dump_datasets.py script.

To download TAACO, run:

```bash
$ make taaco_app
```

Usage:

```bash
$ python scripts/feat_psy_taaco.py --name_or_path "SetFit/20_newsgroups" --split "train"
```

Unfornutately, TAACO does not support running in parallel, so be patient.

"""

import argparse
import os
import sys
from pathlib import Path

from decouple import config

DOCUMENTS_BASEDIR = os.path.join("data", "documents")
FEATURES_BASEDIR = os.path.join("data", "features")
BASE_DIR = Path(__file__).resolve().parent.parent

config = {
    "sourceKeyOverlap": False,
    "sourceLSA": False,
    "sourceLDA": False,
    "sourceWord2vec": False,
    "wordsAll": True,
    "wordsContent": True,
    "wordsFunction": True,
    "wordsNoun": True,
    "wordsPronoun": True,
    "wordsArgument": True,
    "wordsVerb": True,
    "wordsAdjective": True,
    "wordsAdverb": True,
    "overlapSentence": True,
    "overlapParagraph": True,
    "overlapAdjacent": True,
    "overlapAdjacent2": True,
    "otherTTR": True,
    "otherConnectives": True,
    "otherGivenness": True,
    "overlapLSA": True,
    "overlapLDA": True,
    "overlapWord2vec": True,
    "overlapSynonym": True,
    "overlapNgrams": True,
    "outputTagged": False,
    "outputDiagnostic": False,
}


parser = argparse.ArgumentParser(
    description="Extract features from documents using TextStat."
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
    required=False,
    help="The name of the split.",
    default="train",
)

parser.add_argument(
    "--processes",
    dest="processes",
    type=int,
    required=False,
    help="The number of processes to use.",
    default=4,
)

parser.add_argument(
    "--taaco",
    dest="taaco",
    type=str,
    required=False,
    default=os.path.join(BASE_DIR, "taaco"),
)


def run_dataset(name_or_path, split, run_taaco):
    """
    Run TAACO on a single dataset
    """

    source_dir = os.path.join("..", DOCUMENTS_BASEDIR, name_or_path, split)
    target_csv = os.path.join("..", FEATURES_BASEDIR, name_or_path, split, "taaco.csv")

    os.makedirs(os.path.dirname(target_csv), exist_ok=True)

    run_taaco(
        indir=source_dir,
        outdir=target_csv,
        varDict=config,
    )


def feature_extraction(taaco_path, split, name_or_path):
    """
    Extract features from the dataset using TAACO.
    """
    sys.path.insert(0, taaco_path)

    # pylint: disable=wrong-import-position
    from TAACOnoGUI import runTAACO  # noqa: E402

    os.chdir(taaco_path)

    run_dataset(name_or_path, split, runTAACO)


def run():
    args = parser.parse_args()

    feature_extraction(args.taaco, args.split, args.name_or_path)


if __name__ == "__main__":
    run()
