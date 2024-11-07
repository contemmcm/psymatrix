"""
Extract CohMetrix features from a set of documents.

Works with cohmetrixcorecli_1.0.4_linux.i686.deb

Example usage:

```bash
$ python scripts/feat_psy_cohmetrix.py --name_or_path "SetFit/20_newsgroups" --split "train"
```
"""

import argparse
import os
import subprocess
import tempfile
from functools import partial
from glob import glob
from multiprocessing import Pool

import pandas as pd
from decouple import config

COHMETRIX_PATH = config(
    "COHMETRIX_PATH", default="/usr/local/bin/cohmetrixcore/net6.0/CohMetrixCoreCLI"
)
TIMEOUT = config("COHMETRIX_TIMEOUT", cast=int, default=600)

BLACK_LIST_PATH = "cohmetrix.blacklist"

DOCUMENTS_BASEDIR = os.path.join("data", "documents")
FEATURES_BASEDIR = os.path.join("data", "features")

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


def write_blacklist(doc_path):
    """
    Write a document path to the blacklist
    """
    with open(BLACK_LIST_PATH, "a", encoding="utf8") as f:
        f.write(doc_path + "\n")


def load_blacklist():
    """
    Load the blacklist
    """
    if not os.path.exists(BLACK_LIST_PATH):
        return []

    with open(BLACK_LIST_PATH, "r", encoding="utf8") as f:
        return list(set(f.read().split("\n")))


def target_file(doc_path, target_dir):
    """
    Get the target file path for a document
    """
    basename = os.path.basename(doc_path)

    return target_dir, os.path.join(target_dir, basename + ".csv")


def process_text(text: str):
    """
    Process a text using CohMetrix
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        f.close()
        doc_path = f.name

    output_file = process_file(doc_path)

    if not output_file:
        return None

    # read the output file
    with open(output_file, "r", encoding="utf8") as f:
        output = f.read()

    lines = output[1:].split("\n")

    params = {}

    for line in lines:
        if not line:
            continue
        k, v = line.split(",")

        try:
            params[k] = float(v)
        except ValueError:
            params[k] = None

    # Cleanup
    os.remove(doc_path)
    os.remove(output_file)

    return params


def process_file(doc_path: str, target_dir: str = None):
    """
    Process a single file using CohMetrix
    """
    if not target_dir:
        target_dir = os.path.dirname(doc_path)
        output_file = doc_path + ".csv"
    else:
        target_dir, output_file = target_file(doc_path, target_dir)

    command = [COHMETRIX_PATH, doc_path, target_dir]

    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log) as p:
        try:
            p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()

    if os.path.exists(output_file):
        print(f"[OK]\t{output_file}", flush=True)
        return output_file

    print(f"[FAIL]\t{doc_path}", flush=True)

    # write doc_path to blacklist
    write_blacklist(doc_path)

    return None


def process_file_list(pathname: str, target_dir: str, processes: int):
    """
    Process a list of files using CohMetrix
    """
    doc_paths = glob(pathname)

    # Remove files that have already been processed from the queue
    doc_paths = [
        s for s in doc_paths if not os.path.exists(target_file(s, target_dir)[1])
    ]

    # Remove files that are in the blacklist
    blacklist = load_blacklist()
    doc_paths = [s for s in doc_paths if s not in blacklist]

    if not doc_paths:
        print("All files have already been processed.")
        return

    # Sort the files by name
    doc_paths.sort()

    process_file_func = partial(process_file, target_dir=target_dir)

    with Pool(processes) as pool:
        pool.map(process_file_func, doc_paths)


def compile_features(source_dir: str, output_file: str):
    """
    Compile features from multiple files into a single file
    """
    # Get the list of files
    files = glob(os.path.join(source_dir, "*.csv"))

    # Read the files
    data = []

    for file in files:
        features = read_features(file)
        data.append(features)

    df = pd.DataFrame(data)

    # sort by filename
    df = df.sort_values(by="file")

    df.to_csv(output_file, index=False)


def read_features(path):
    with open(path, "r", encoding="utf8") as f:
        lines = [line.strip() for line in f.readlines()]

    features = {
        "file": os.path.basename(path).replace(".csv", ""),
    }

    for line in lines:
        k, v = line.split(",")
        if k.startswith("{"):
            k = k[1:-1]

        try:
            features[k] = float(v)
        except ValueError:
            continue

    return features


def run():

    args = parser.parse_args()

    input_files = os.path.join(
        DOCUMENTS_BASEDIR, args.name_or_path, args.split, "*.txt"
    )
    target_dir_base = os.path.join(FEATURES_BASEDIR, args.name_or_path, args.split)
    target_dir = os.path.join(target_dir_base, "cohmetrix")
    output_file = os.path.join(target_dir_base, "cohmetrix.csv")
    n_threads = args.processes

    print("Input: ", input_files)
    print("Target: ", target_dir)
    print("Threads: ", n_threads)

    # create target base directory
    os.makedirs(target_dir, exist_ok=True)

    process_file_list(input_files, target_dir, n_threads)

    compile_features(target_dir, output_file)


if __name__ == "__main__":

    with open("cohmetrix.log", "a", encoding="utf8") as log:
        run()
