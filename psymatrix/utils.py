import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_selected_metafeatures(
    datasets,
    fname_selection,
    groups=("cohmetrix", "lang", "mfe", "taaco", "textstat", "topic_lda"),
    split="train",
    normalize=False,
    fillna=None,
    include_datasets_ids=False,
):
    """
    Load metafeatures from the specified groups for the specified split from
    the data/metafeatures directory.
    """
    if isinstance(datasets, str):
        datasets = load_datasets(datasets)

    metafeatures = []

    for dataset_id in datasets:
        if include_datasets_ids:
            row = {"dataset": dataset_id}
        else:
            row = {}

        for group in groups:
            fname = f"data/metafeatures/{dataset_id}/{split}/{group}.json"
            with open(fname, "r", encoding="utf8") as f:
                row.update(json.load(f))

        metafeatures.append(row)

    metafeatures = pd.DataFrame(metafeatures)

    with open(fname_selection, "r", encoding="utf8") as f:
        selected_features = [line.strip() for line in f.readlines() if line.strip()]
    metafeatures = metafeatures[selected_features]

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        metafeatures = pd.DataFrame(
            scaler.fit_transform(metafeatures), columns=metafeatures.columns
        )

    if fillna is not None:
        metafeatures.fillna(fillna, inplace=True)

    return metafeatures


def load_metafeatures(
    datasets,
    groups=("cohmetrix", "lang", "mfe", "taaco", "textstat", "topic_lda"),
    split="train",
    config=None,
    normalize=False,
    fillna=None,
    include_datasets_ids=False,
):
    """
    Load metafeatures from the specified groups for the specified split from
    the data/metafeatures directory.
    """

    if isinstance(datasets, str):
        datasets = load_datasets(datasets)

    metafeatures = []

    for dataset_id in datasets:
        if include_datasets_ids:
            row = {"dataset": dataset_id}
        else:
            row = {}

        total = 0

        for group in groups:
            fname = f"data/metafeatures/{dataset_id}/{split}/{group}.json"
            with open(fname, "r", encoding="utf8") as f:
                data = json.load(f)
                total += len(data.keys())
                row.update(data)

        print(total, dataset_id)
        metafeatures.append(row)

    metafeatures = pd.DataFrame(metafeatures)  # 124, 143, 145

    if config:
        with open(f"metafeatures.{config}.config", "r", encoding="utf8") as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        metafeatures = metafeatures[selected_features]

    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        metafeatures = pd.DataFrame(
            scaler.fit_transform(metafeatures), columns=metafeatures.columns
        )

    if fillna is not None:
        metafeatures.fillna(fillna, inplace=True)

    return metafeatures


def load_datasets(fname):
    """
    Load a list of dataset IDs from the specified file.
    """
    with open(fname, "r", encoding="utf8") as f:
        datasets = [
            line.strip() for line in f.readlines() if not line.strip().startswith("#")
        ]

    return datasets


def load_models(fname):
    """
    Load a list of model IDs from the specified file.
    """
    with open(fname, "r", encoding="utf8") as f:
        models = [line.strip() for line in f.readlines()]

    return models


def load_performance(model_id, datasets, performance_type="accuracy_norm"):
    rows = []

    for dataset_id in datasets:
        fname = f"data/performance/{dataset_id}/test/performance.json"
        with open(fname, "r", encoding="utf8") as f:
            perf = json.load(f)

        rows.append(
            {"dataset": dataset_id, performance_type: perf[model_id][performance_type]}
        )
    df = pd.DataFrame(rows)

    return df
