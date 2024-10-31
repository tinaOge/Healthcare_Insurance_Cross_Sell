import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def elbow_method(X_cluster):
    """Determine the optimal number of clusters using the Elbow method."""
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_cluster)
        inertia.append(kmeans.inertia_)


    # Plot the Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    optimal_k = 6  # Replace with your logic to determine the elbow point
    return optimal_k



def fit_kmeans(X_cluster, optimal_k):
    """Fit KMeans model and return cluster labels."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_cluster)
    return kmeans.labels_


def visualize_clusters(df_cluster):
    """Visualize cluster counts and key features for each cluster."""
    cluster_counts = df_cluster['Cluster'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    plt.title("Cluster Counts")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()

    # Identify key features with the highest values in each cluster
    cluster_means = df_cluster.groupby('Cluster').mean()
    for cluster in cluster_means.index:
        highest_mean_features = cluster_means.loc[cluster].nlargest(2)
        print(f"\nCluster {cluster} Key Features:")
        print(highest_mean_features)


