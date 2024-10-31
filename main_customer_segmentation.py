import pandas as pd
from scripts.customer_segmentation.eda import load_data, perform_eda
from scripts.customer_segmentation.data_preprocessing import preprocess_data
from scripts.customer_segmentation.clustering import elbow_method, fit_kmeans, visualize_clusters
from scripts.customer_segmentation.evaluation import evaluate_clustering


def main():
    # Step 1: Load data
    df = load_data('data/raw/train.csv')  # Ensure the path to your data is correct
    print("Data loaded successfully.")

    # Step 2: Perform Exploratory Data Analysis (EDA)
    print("Performing EDA...")
    shape, numeric_summary, categorical_summary = perform_eda(df)
    print("Data shape:", shape)
    print("Data Description:\n", numeric_summary)
    print("Categorical Description:\n", categorical_summary)

    # Step 3: Preprocess data
    print("Preprocessing data...")
    df_cluster, y, selected_columns = preprocess_data(df)  # Capture selected_columns
    print("Data preprocessing completed. New head:", df_cluster.head())

    # Step 4: Define X_cluster
    X_cluster = df_cluster[selected_columns]

    print("Determining optimal number of clusters using the Elbow method...")
    optimal_k = elbow_method(X_cluster)

    # Step 5: Fit KMeans model
    print("Fitting KMeans model...")
    cluster_labels = fit_kmeans(X_cluster, optimal_k)
    df_cluster['Cluster'] = cluster_labels
    visualize_clusters(df_cluster)

    print(f"Clustering completed with optimal k: {optimal_k}")

    # Step 6: Evaluate clustering performance
    print("Evaluating clustering...")
    evaluate_clustering(df_cluster, df_cluster['Cluster'], optimal_k)


if __name__ == "__main__":
    main()