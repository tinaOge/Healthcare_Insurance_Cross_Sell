import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X_cluster, cluster_labels, optimal_k):
    """Evaluate clustering performance using Silhouette Score and Davies-Bouldin Index."""
    # Calculate Silhouette Score
    sil_score = silhouette_score(X_cluster, cluster_labels)
    print(f'Silhouette Score for {optimal_k} clusters: {sil_score}')

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(X_cluster, cluster_labels)
    print(f'Davies-Bouldin Index: {db_index}')