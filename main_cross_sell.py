import pandas as pd
from scripts.cross_sell_prediction.eda import perform_eda
from scripts.cross_sell_prediction.data_preprocessing import engineer_features, split_data, preprocess_data, balance_data
from scripts.cross_sell_prediction.visualization import plot_target_distribution, plot_feature_distributions, \
    plot_histograms, plot_boxplots, plot_categorical_vs_target
from scripts.cross_sell_prediction.model_training_and_evaluation import train_and_evaluate_model, generate_submission
def load_data():
    # Load your train and test data here
    df_train = pd.read_csv('data/raw/train.csv', index_col='id')
    df_test = pd.read_csv('data/raw/test.csv', index_col='id')
    return df_train, df_test

def main():
    # Load data
    df_train, df_test = load_data()

    # Perform visualizations as needed
    plot_target_distribution(df_train)
    plot_feature_distributions(df_train)
    plot_histograms(df_train)
    plot_boxplots(df_train)
    plot_categorical_vs_target(df_train)

    # Perform EDA
    combined_df, numeric_summary, categorical_summary = perform_eda(df_train, df_test)
    print("\nNumeric Summary:\n", numeric_summary)
    print("\nCategorical Summary:\n", categorical_summary)

    # Engineer features
    combined_df = engineer_features(combined_df)

    # Split data
    X_train, X_test, y_train, y_test, df_test_final = split_data(combined_df)

    # Preprocess data
    X_train_processed, X_test_processed, df_test_processed, preprocessor = preprocess_data(X_train, X_test, df_test_final)

    # Balance training data
    X_train_balanced, y_train_balanced = balance_data(X_train_processed, y_train)

    # Train and evaluate the model
    best_model = train_and_evaluate_model(X_train_balanced, y_train_balanced, X_test_processed, y_test)

    # Generate and save submission
    generate_submission(best_model, df_test_processed, df_test_final, output_path='submission.csv')


if __name__ == "__main__":
    main()
