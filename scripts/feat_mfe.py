"""
Extract complexity features from classification datasets, considering as features
the TF-IDF vec of the documents, and as target the labels provided by the dataset.

Example usage:

```bash
$ python scripts/feat_mfe.py --name_or_path "SetFit/20_newsgroups" --split "train"
```
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from datasets import load_dataset
from pymfe.mfe import MFE
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_MAX_FEATURES = 200

META_FEATURES_BASEDIR = os.path.join("data", "metafeatures")

GROUPS = {
    "clustering": ("ch", "int", "nre", "pb", "sc", "sil", "vdb", "vdu"),
    "concept": ("cohesiveness", "conceptvar", "impconceptvar", "wg_dist"),
    "info-theory": (
        "attr_conc",
        "attr_ent",
        "class_conc",
        "class_ent",
        "eq_num_attr",
        "joint_ent",
        "mut_inf",
        "ns_ratio",
    ),
    "landmarking": (
        "best_node",
        "elite_nn",
        "linear_discr",
        "naive_bayes",
        "one_nn",
        "random_node",
        "worst_node",
    ),
    "general": (
        "attr_to_inst",
        "cat_to_num",
        "freq_class",
        "inst_to_attr",
        "nr_attr",
        "nr_bin",
        "nr_cat",
        "nr_class",
        "nr_inst",
        "nr_num",
        "num_to_cat",
    ),
    "statistical": (
        "can_cor",
        "cor",
        "cov",
        "eigenvalues",
        "g_mean",
        "gravity",
        "h_mean",
        "iq_range",
        "kurtosis",
        "lh_trace",
        "mad",
        "max",
        "mean",
        "median",
        "min",
        "nr_cor_attr",
        "nr_disc",
        "nr_norm",
        "nr_outliers",
        "p_trace",
        "range",
        "roy_root",
        "sd",
        "sd_ratio",
        "skewness",
        "sparsity",
        "t_mean",
        "var",
        "w_lambda",
    ),
    "complexity": (
        "c1",
        "c2",
        "cls_coef",
        "density",
        "f1",
        "f1v",
        "f2",
        "f3",
        "f4",
        "hubs",
        "l1",
        "l2",
        "l3",
        "lsc",
        "n1",
        "n2",
        "n3",
        "n4",
        "t1",
        "t2",
        "t3",
        "t4",
    ),
}

SUMMARIES = (
    "mean",
    "nanmean",
    "sd",
    "nansd",
    "var",
    "nanvar",
    "count",
    "nancount",
    "histogram",
    "nanhistogram",
    "iq_range",
    "naniq_range",
    "kurtosis",
    "nankurtosis",
    "max",
    "nanmax",
    "median",
    "nanmedian",
    "min",
    "nanmin",
    "quantiles",
    "nanquantiles",
    "range",
    "nanrange",
    "skewness",
    "nanskewness",
    "sum",
    "nansum",
    "powersum",
    "pnorm",
    "nanpowersum",
    "nanpnorm",
)

parser = argparse.ArgumentParser(
    description="Extract diverse meta-features from classifcation dataset."
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


class MFECodec:

    def __init__(self, tfidf, labels):
        self.tfidf = tfidf.todense()
        self.labels = labels

    def eval_group(self, group: str):

        mfe = MFE(groups=[group], summary=SUMMARIES)
        mfe.fit(self.tfidf, self.labels, verbose=0)

        ft = mfe.extract(verbose=1)
        ft_keys = [f"{group}:{f}" for f in ft[0]]

        return dict(zip(ft_keys, ft[1]))


def load_documents(name_or_path, split):
    """
    Load documents from huggingface dataset
    """
    dataset = load_dataset(name_or_path)

    ds_split = dataset[split]

    return ds_split


def process_dataset(name_or_path: str, split: str, max_features: int):
    """
    Extract metafeatures from a single classification dataset.
    """
    print(f"Extracting metafeatures for {name_or_path} ({split})...")

    metafeatures = evaluate_dataset(name_or_path, split, max_features=max_features)

    # Save metafeatures to a json file
    fname = os.path.join(META_FEATURES_BASEDIR, name_or_path, split, "mfe.json")

    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, "w", encoding="utf8") as f:
        json.dump(metafeatures, f, indent=2, sort_keys=True)

    return metafeatures


def evaluate_dataset(name_or_path: str, split: str, **kwargs):
    """
    Calculate the imbalance ratio of a dataset.
    """
    corpus = load_documents(name_or_path, split)
    labels = np.array(corpus["label"], dtype=int)
    groups = list(GROUPS.keys())

    # convert None to empty string
    corpus = [doc if doc is not None else "" for doc in corpus["text"]]

    vectorizer = TfidfVectorizer(**kwargs)
    tfidf = vectorizer.fit_transform(corpus)

    codec = MFECodec(tfidf, labels)

    with Pool(len(groups)) as pool:
        _feats = pool.map(codec.eval_group, groups)

    feats = {k: v for f in _feats for k, v in f.items()}

    # feats = {}
    # for group in GROUPS:
    #     feats.update(codec.eval_group(group))

    # Fix object types
    feats["general:nr_bin"] = int(feats["general:nr_bin"])
    feats["statistical:nr_outliers"] = int(feats["statistical:nr_outliers"])
    feats["clustering:sc"] = int(feats["clustering:sc"])

    # Replace NaN for None for JSON serialization
    for k in feats:
        if np.isnan(feats[k]):
            feats[k] = None

    # Add prefix to keys
    feats = {f"mfe:{k}": v for k, v in feats.items()}

    # Convert numpy types (int64, float64, etc) to native python types
    feats = {k: v.item() if isinstance(v, np.generic) else v for k, v in feats.items()}

    return feats


def run():
    args = parser.parse_args()

    metafeatures = process_dataset(
        args.name_or_path, args.split, max_features=TFIDF_MAX_FEATURES
    )
    print(metafeatures)


if __name__ == "__main__":
    run()
