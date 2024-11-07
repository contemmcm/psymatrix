import json

import numpy as np
import pandas as pd

from psymatrix.experiments import load_datasets, load_models, load_performance

loss_all = load_performance("emnlp24", "loss", normalized=True)
accuracy_all = load_performance("emnlp24", "accuracy", normalized=True)


def extract_prob_topk(predictions, targets):
    """
    Calculates the probabilty of selecting the optimal model in the top-k.
    """
    topk_vec = []

    for k in range(len(predictions)):
        if targets[0] in predictions[: (k + 1)]:
            topk_vec.append(1)
        else:
            topk_vec.append(0)

    return topk_vec


def convert_rank2perf(rank, dataset_idx):

    perf = loss_all[dataset_idx]
    bestk_vec = []
    for k in range(len(rank)):
        best_k = perf[rank[0 : k + 1]].max()
        bestk_vec.append(best_k)

    return bestk_vec


def eval_perf_random(experiment):

    loss_all = load_performance(experiment, "loss", normalized=True)

    n_repeats = 500
    n_models = len(loss_all)
    n_datasets = len(loss_all.columns)

    rows = []

    for ds in range(n_datasets):
        print(".", end="")
        for _ in range(n_repeats):
            random_rank = np.random.permutation(n_models)
            random_perf = convert_rank2perf(random_rank, ds)
            rows.append(random_perf)

    df = pd.DataFrame(rows)

    return df.mean(axis=0).to_numpy()


def print_as_matlab(
    df_prob_psy: pd.DataFrame,
    df_perf_psy: pd.DataFrame,
    df_prob_naive: pd.DataFrame,
    df_perf_naive: pd.DataFrame,
):
    prob_psy_mean = df_prob_psy.mean(axis=0).tolist()
    prob_naive_mean = df_prob_naive.mean(axis=0).tolist()

    perf_psy_mean = df_perf_psy.mean(axis=0).tolist()
    perf_naive_mean = df_perf_naive.mean(axis=0).tolist()

    print("%% Probability of selecting the optimal model in the top-k")
    print(f"prob_psy_mean = {prob_psy_mean};")
    print(f"prob_naive_mean = {prob_naive_mean};")

    print("%% Evolution of mean performance over the top-k")
    print(f"perf_psy_mean = {perf_psy_mean};")
    print(f"perf_naive_mean = {perf_naive_mean};")


def eval(experiment, output="stdout"):
    models = load_models(experiment)
    datasets = load_datasets(experiment)
    results = json.loads(open("data/neural_net/repeat_0_nnet.json").read())

    num_folds = len(results.keys())

    rows_probabilities_psymatrix = []
    rows_performances_psymatrix = []

    rows_probabilities_naive = []
    rows_performances_naive = []
    naive_rank = loss_all.mean(axis=1).to_numpy().argsort()[::-1]

    for k in range(1, num_folds + 1):

        fold = results[f"fold_{k}"]

        test_ids = fold["test_ids"]  # Datasets used for testing

        for i, ds in enumerate(test_ids):
            predictions = fold["y_pred_rank"][i]
            targets = fold["y_real_rank"][i]

            rows_probabilities_psymatrix.append(extract_prob_topk(predictions, targets))
            rows_performances_psymatrix.append(convert_rank2perf(predictions, ds))

            rows_probabilities_naive.append(extract_prob_topk(naive_rank, targets))
            rows_performances_naive.append(convert_rank2perf(naive_rank, ds))

    df_prob_psy = pd.DataFrame(rows_probabilities_psymatrix)
    df_perf_psy = pd.DataFrame(rows_performances_psymatrix)

    df_prob_naive = pd.DataFrame(rows_probabilities_naive)
    df_perf_naive = pd.DataFrame(rows_performances_naive)

    if output == "stdout":
        print_as_matlab(df_prob_psy, df_perf_psy, df_prob_naive, df_perf_naive)
        return None

    if output in ("prob", "probability", "probabilities"):
        return df_prob_psy.mean(axis=0).tolist(), df_prob_naive.mean(axis=0).tolist()

    if output in ("perf", "performance"):
        return df_perf_psy.mean(axis=0).tolist(), df_perf_naive.mean(axis=0).tolist()

    raise ValueError(f"Unknown output format: {output}")


if __name__ == "__main__":
    eval("emnlp24")
