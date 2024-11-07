import json

from psymatrix.utils import load_metafeatures


def run():
    data = load_metafeatures("emnlp24_datasets.txt")

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

    # for feat in stats.keys():
    #     json.dumps(stats[feat], indent=4)

    with open("metafeatures_stats.json", "w", encoding="utf8") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    run()
