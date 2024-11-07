"""
Detect the language of the documents in the dataset.

Example usage:

```bash
$  python -m scripts.feat_lang --name_or_path "SetFit/20_newsgroups" --split "train"
```
"""

import argparse
import os
from multiprocessing import Pool

import pandas as pd
from langdetect import PROFILES_DIRECTORY, DetectorFactory, LangDetectException

from scripts.feat_psy_textstat import get_documents

FEATURES_BASEDIR = os.path.join("data", "features")

DetectorFactory.seed = 0

factory = DetectorFactory()
factory.load_profile(PROFILES_DIRECTORY)


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


def detect_languages_single(text):
    """
    Detect the language of a single document.
    """
    print(".", end="")
    detector = factory.create()
    detector.set_max_text_length(100000)  # 100 KB

    detector.append(text)

    features = {k: 0.0 for k in detector.langlist}

    try:
        for p in detector.get_probabilities():
            features[p.lang] = p.prob
    except LangDetectException as err:
        print(f"{err}: {text}")

    return features


def detect_languages(documents, processes=4):
    """
    Detect the language of a a set of documents.
    """
    with Pool(processes) as pool:
        doc_probabilities = pool.map(detect_languages_single, documents)

    return doc_probabilities


def save_csv(name_or_path, split, features_per_document):

    df = pd.DataFrame(features_per_document)

    # Ensure the cache directory exists
    output_file = os.path.join(FEATURES_BASEDIR, name_or_path, split, "lang.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # sort rows by file name
    df = df.sort_values(by="file")

    df.to_csv(output_file, index=False)


def run():

    args = parser.parse_args()

    documents, paths = get_documents(args.name_or_path, args.split)
    features = detect_languages(documents, processes=args.processes)
    features = [{**{"file": os.path.basename(p)}, **f} for p, f in zip(paths, features)]
    save_csv(args.name_or_path, args.split, features)


if __name__ == "__main__":
    run()
