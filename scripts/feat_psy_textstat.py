"""
Extract textstat features from a set of documents.

Example usage:
    
```bash
$ python scripts/feat_psy_textstat.py --name_or_path "SetFit/20_newsgroups" --split "train"
```

"""

import argparse
import glob
import os
from multiprocessing import Pool

import pandas as pd
import textstat

DOCUMENTS_BASEDIR = os.path.join("data", "documents")
FEATURES_BASEDIR = os.path.join("data", "features")

FEATURE_NAMES = [
    "automated_readability_index",
    "avg_character_per_word",
    "avg_letter_per_word",
    "avg_sentence_length",
    "avg_sentence_per_word",
    "avg_syllables_per_word",
    "char_count",
    "coleman_liau_index",
    "count_faseeh",
    "dale_chall_readability_score",
    "dale_chall_readability_score_v2",
    "difficult_words",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "gunning_fog",
    "letter_count",
    "lexicon_count",
    "linsear_write_formula",
    "lix",
    "long_word_count",
    "mcalpine_eflaw",
    "miniword_count",
    "monosyllabcount",
    "polysyllabcount",
    "reading_time",
    "rix",
    "sentence_count",
    "smog_index",
    "spache_readability",
    "syllable_count",
    "words_per_sentence",
]

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


def get_documents(name_or_path, split):
    """
    Load documents from disk in .cache/datasets/
    """
    base_dir = os.path.join(DOCUMENTS_BASEDIR, name_or_path, split)
    documents_paths = glob.glob(base_dir + "/*.txt")
    documents = []
    for doc_path in documents_paths:
        with open(doc_path, "r", encoding="utf8") as f:
            documents.append(f.read())
    return documents, documents_paths


def textstat_features(name_or_path, split, processes: int = 4):
    """
    Extract textstat features from a set of documents.
    """
    documents, paths = get_documents(name_or_path, split)

    with Pool(processes) as pool:
        features_per_document = pool.map(extract_textstat_features_single, documents)

    return features_per_document, paths


def extract_textstat_features_single(text):
    """
    Extract textstat features from a single document.
    """
    print(".", end="")
    return [_feature(feature_name, text) for feature_name in FEATURE_NAMES]


def save_csv(name_or_path, features_per_document, paths, split):
    rows = []

    for i in range(len(features_per_document)):
        features = {k: v for k, v in zip(FEATURE_NAMES, features_per_document[i])}
        features["file"] = os.path.basename(paths[i])
        rows.append(features)

    df = pd.DataFrame(rows)

    output_file = os.path.join(FEATURES_BASEDIR, name_or_path, split, "textstat.csv")

    # Ensure the cache directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Sort rows by filename
    df = df.sort_values(by="file")

    df.to_csv(output_file, index=False)


def _feature(feature_name, text):
    """
    Extract a single textstat feature from a single document.
    """
    if text is None:
        text = ""

    try:
        return getattr(textstat, feature_name)(text)
    except ZeroDivisionError:
        return 0.0


def run():

    args = parser.parse_args()

    features_per_document, paths = textstat_features(
        args.name_or_path, split=args.split, processes=args.processes
    )

    save_csv(args.name_or_path, features_per_document, paths, split=args.split)


if __name__ == "__main__":
    run()
