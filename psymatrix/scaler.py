import json
import os

import pandas as pd

from psymatrix.utils import load_metafeatures

pd.options.mode.copy_on_write = True


class Scaler:
    """
    Class to scale new metafeature readings based on the statistics of the training
    data.
    """

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.path = self._build_path(experiment)

        with open(self.path, "r", encoding="utf8") as f:
            self.stats = json.load(f)

    def transform(self, data):
        """
        Scale the data based on the statistics of the training data.
        """
        if isinstance(data, pd.Series):
            for name in data.index:
                data[name] -= self.stats[name]["min"]
                data[name] /= self.stats[name]["max"] - self.stats[name]["min"]
        else:
            raise ValueError("Data must be a pandas Series.")

        return data

    @staticmethod
    def fit(experiment: str):
        """
        Save the scaler statistics to a file.
        """
        ds_path = os.path.join("experiments", experiment, "datasets.txt")
        output_path = Scaler._build_path(experiment)

        data = load_metafeatures(ds_path)

        stats = {}

        for feat in data.columns:
            stats[feat] = {
                "mean": data[feat].mean(),
                "nanmean": data[feat].mean(skipna=True),
                "std": data[feat].std(),
                "nanstd": data[feat].std(skipna=True),
                "min": float(data[feat].min()),
                "max": float(data[feat].max()),
                "count": int(data[feat].count()),
                "nancount": int(data[feat].isna().sum()),
            }

        with open(output_path, "w", encoding="utf8") as f:
            json.dump(stats, f, indent=4)

    @staticmethod
    def _build_path(experiment: str):
        return os.path.join("experiments", experiment, "scaler.json")
