from psymatrix.metafeatures.selection import k_means, pre_processing
from psymatrix.utils import load_metafeatures


def run(num_features: int = 625):
    metafeatures = load_metafeatures(
        "emnlp24_datasets.txt", include_datasets_ids=False, fillna=0.0
    )

    metafeatures = pre_processing.drop_constant_features(metafeatures)

    metafeatures = pre_processing.drop_correlated_features(metafeatures, threshold=0.95)

    k_means.dimensionality_reduction(
        metafeatures,
        num_clusters=num_features,
        fout=f"metafeatures.{num_features}.config",
    )


if __name__ == "__main__":
    run()
