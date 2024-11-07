import multiprocessing
import os
import sys
from functools import partial

from scripts.feat_psy_taaco import run_dataset

N_TREADS = 2


def load_datasets():
    """
    Load datasets from emnlp24_datasets.txt
    """
    with open("emnlp24_datasets.txt", encoding="utf8") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]


def run():
    """
    Run TAACO on all datasets
    """

    datasets = load_datasets()

    taaco_path = "taaco"

    sys.path.insert(0, taaco_path)

    # pylint: disable=wrong-import-position
    from TAACOnoGUI import runTAACO  # noqa: E402

    os.chdir(taaco_path)

    t = partial(run_dataset, split="train", run_taaco=runTAACO)

    with multiprocessing.Pool(N_TREADS) as pool:
        pool.map(t, datasets)


if __name__ == "__main__":
    run()
