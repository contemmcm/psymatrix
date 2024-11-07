"""
Extract topic features from a set of documents.

Example usage:

```bash
$ python scripts/feat_topic_lda.py --name_or_path "SetFit/20_newsgroups" --split "train"
```

"""

import argparse
import os
import re
from glob import glob
from random import shuffle

import pandas as pd
from datasets import load_dataset
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FEATURES_BASEDIR = os.path.join("data", "features")

LDA_BASEDIR = os.path.join("data", "topics", "lda")
LDA_MODEL_FILE = os.path.join(LDA_BASEDIR, "lda.gensim")
DICTIONARY_FILE = os.path.join(LDA_BASEDIR, "lda.dict")
NUM_TOPICS = 100

parser = argparse.ArgumentParser(
    description="Extract topic features from documents using LDA."
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


def load_corpus(namespace: str, split: str = "train"):
    documents_paths = glob(f"data/documents/{namespace}/*/{split}/*.txt")
    fname = os.path.join(LDA_BASEDIR, f"{namespace}_{split}_corpus.txt")

    if os.path.exists(fname):
        with open(fname, "r", encoding="utf8") as fin:
            return [line.strip().split() for line in fin if line.strip()]

    docs = []

    for doc_path in documents_paths:
        with open(doc_path, "r", encoding="utf8") as fin:
            text = fin.read()

            if text.strip():
                docs.append(parse_document(text.strip()))

    # Save the corpus

    with open(fname, "w", encoding="utf8") as fout:
        for doc in docs:
            if doc:
                fout.write(" ".join(doc) + "\n")

    return docs


def load_documents(name_or_path, split):
    """
    Load documents from huggingface dataset
    """
    dataset = load_dataset(name_or_path)

    ds_split = dataset[split]

    return ds_split


def parse_document(text, lowercase=True):
    """
    Parse a document into a list of words.
    """
    if not text:
        return []

    if lowercase:
        text = text.lower()

    # extract only the words
    words = re.findall(r"\w+", text)

    # remove stopwords
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    # remove short words
    words = [word for word in words if len(word) > 2]

    # remove numbers
    words = [word for word in words if not word.isnumeric()]

    return words


def get_topics(document, dictionary, lda_model):
    num_topics = lda_model.num_topics
    doc_bow = dictionary.doc2bow(parse_document(document))
    doc_topics = lda_model.get_document_topics(doc_bow)

    topic_dist = [0.0] * num_topics

    for topic, prob in doc_topics:
        topic_dist[topic] = prob

    return {f"topic_{k}": v for k, v in enumerate(topic_dist)}


def process_dataset(
    name_or_path: str, split: str, dictionary: Dictionary, lda_model: LdaModel
):
    """
    Process a dataset
    """
    documents = load_documents(name_or_path, split)
    num_max_digits = len(str(documents.num_rows))

    rows = []
    for i, document in enumerate(documents["text"]):
        topics = get_topics(document, dictionary, lda_model)
        topics["file"] = str(i).zfill(num_max_digits) + ".txt"
        rows.append(topics)

    df = pd.DataFrame(rows)

    fname = os.path.join(FEATURES_BASEDIR, name_or_path, split, "topic_lda.csv")

    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # sort rows by filename
    df = df.sort_values(by="file")

    df.to_csv(fname, index=False)

    print(fname)


def run():
    """
    Run CohMetrix on all datasets
    """
    args = parser.parse_args()

    dictionary = Dictionary.load(DICTIONARY_FILE)
    lda_model = LdaModel.load(LDA_MODEL_FILE)

    process_dataset(args.name_or_path, args.split, dictionary, lda_model)


if __name__ == "__main__":
    run()
