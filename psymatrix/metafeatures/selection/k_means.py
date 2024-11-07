import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def dimensionality_reduction(data, num_clusters=500, fout=None):

    # Standardize the data
    scaler = StandardScaler()

    data_transposed = data.T

    data_scaled = scaler.fit_transform(data_transposed)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data_scaled)

    # Create a DataFrame with the cluster assignments
    clusters = pd.DataFrame(
        data=kmeans.labels_, index=data.columns, columns=["cluster"]
    )

    feature_importance = pd.DataFrame()
    for cluster in range(num_clusters):
        cluster_features = clusters[clusters["cluster"] == cluster].index
        cluster_data = data[cluster_features]
        feature_variances = cluster_data.var().sort_values(ascending=False)
        feature_importance = pd.concat(
            [feature_importance, feature_variances.head(1).rename(f"Cluster_{cluster}")]
        )

    # Extract the list of selected features
    selected_features = feature_importance.index.tolist()

    if fout:
        with open(fout, "w", encoding="utf8") as f:
            for i, feature in enumerate(selected_features):
                f.write(f"{feature}")
                if i < len(selected_features) - 1:
                    f.write("\n")

    return selected_features
