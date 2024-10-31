import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix


def train_and_evaluate_model(
        X_train_balanced,
        y_train_balanced,
        X_test_processed,
        y_test):
    # Define model with specified hyperparameters
    best_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        subsample=1.0,
        n_estimators=200,
        max_depth=8,
        learning_rate=0.2,
        gamma=0,
        colsample_bytree=1.0
    )

    # Train model on balanced training data
    best_model.fit(X_train_balanced, y_train_balanced)

    # Predict on test data
    y_pred = best_model.predict(X_test_processed)
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

    # Calculate ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return best_model


def generate_submission(
        best_model,
        df_test_processed,
        df_test,
        output_path='submission.csv'):
    # Generate predictions and probabilities for test data
    predictions_proba = best_model.predict_proba(df_test_processed)[:, 1]
    predictions = best_model.predict(df_test_processed)

    # Prepare submission DataFrame
    submission = pd.DataFrame({
        'id': df_test.index,
        'Response': predictions,
        'Predicted Probability': predictions_proba
    })

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(submission.head())
