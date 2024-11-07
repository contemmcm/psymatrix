"""
Converts document features to dataset metafeatures.

Example usage:

```bash
$  python -m scripts.metafeatures --split "train" --feature textstat --name_or_path "SetFit/20_newsgroups" 
```
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

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
    "--feature",
    dest="feature",
    type=str,
    required=True,
    help="The name of the feature to summarize into metafeatures.",
)


def summaries(samples, prefix=""):
    """
    Compute metafeatures from samples of a given distribution.
    """

    if prefix:
        prefix += ":"

    metafeatures = {
        f"{prefix}corrcoef": np.corrcoef(samples),
        f"{prefix}iqr": stats.iqr(samples),
        f"{prefix}kstatvar": stats.kstatvar(samples),
        f"{prefix}kurtosis": stats.kurtosis(samples),
        f"{prefix}max": float(stats.tmax(samples)),
        f"{prefix}mean": np.mean(samples),
        f"{prefix}mad": stats.median_abs_deviation(samples),
        f"{prefix}min": float(stats.tmin(samples)),
        f"{prefix}mode": float(stats.mode(samples, keepdims=False).mode),
        f"{prefix}mode_count": int(stats.mode(samples, keepdims=False).count),
        f"{prefix}moment": stats.moment(samples),
        f"{prefix}ptp": float(np.ptp(samples)),
        f"{prefix}q1": np.quantile(samples, 0.25),
        f"{prefix}q2": np.quantile(samples, 0.50),  # median
        f"{prefix}q3": np.quantile(samples, 0.75),
        f"{prefix}sem": stats.sem(samples),
        f"{prefix}skew": stats.skew(samples),
        f"{prefix}std": np.std(samples),
        f"{prefix}var": np.var(samples),
        f"{prefix}variation": stats.variation(samples),
    }

    # replace NaN values with 0
    for key, value in metafeatures.items():
        if np.isnan(value):
            metafeatures[key] = 0.0

    return metafeatures


def extract_metafeatures(data, prefix):
    """
    Summarize the samples for each feature using several statistics summary functions,
    such as mean, median, standard deviation, etc. The result is a dictionary with the
    feature names as keys and the summary statistics as values.
    """

    result = {}

    for i in data.columns:
        feature_name = str(i)
        samples = data[i]
        samples_metafeatures = summaries(samples, f"{prefix}:{feature_name}")
        result.update(samples_metafeatures)

    return result


def read_features(name_or_path, split, feature):
    df = pd.read_csv(f"data/features/{name_or_path}/{split}/{feature}.csv")

    # Drop the file column
    if "file" in df.columns:
        df = df.drop(columns=["file"])

    if "Filename" in df.columns:
        df = df.drop(columns=["Filename"])

    return df


def write_json(metafeatures, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(metafeatures, f, indent=2)


def run():
    args = parser.parse_args()

    df = read_features(args.name_or_path, args.split, args.feature)

    metafeatures = extract_metafeatures(df, prefix=args.feature)
    write_json(
        metafeatures,
        f"data/metafeatures/{args.name_or_path}/{args.split}/{args.feature}.json",
    )


if __name__ == "__main__":
    run()
