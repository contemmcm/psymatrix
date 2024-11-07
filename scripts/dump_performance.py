import pandas as pd

from psymatrix import utils

metric = "accuracy_norm"


def run():
    datasets = utils.load_datasets("emnlp24_datasets.txt")
    models = utils.load_models("emnlp24_models.txt")

    # Save to a text file

    with open("performance.txt", "w") as f:

        for m, model in enumerate(models):
            performances = utils.load_performance(model, datasets, metric)

            for v, val in enumerate(performances[metric].to_numpy()):
                f.write(f"{val:0.5f}")

                if v < len(performances) - 1:
                    f.write(" ")

            if m < len(models) - 1:
                f.write("\n")

    # Open the text file and read the values into a DataFrame
    df = pd.read_csv("performance.txt", sep=" ", header=None)

    # Rank the models based on their performance
    df_rank = df.rank(axis=0, method="max", ascending=False).astype(int)

    # Save the rankings to a text file
    df_rank.to_csv("ranking.txt", sep=" ", header=False, index=False)


if __name__ == "__main__":
    run()
