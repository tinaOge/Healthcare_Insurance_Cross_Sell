# scripts/cross_sell_prediction/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style globally for all plots
sns.set(style="whitegrid")


def plot_target_distribution(df):
    """
    Create a count plot for the target feature.

    Parameters:
    - df (DataFrame): DataFrame containing the target column 'Response'.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Response', hue='Response', data=df, palette='Set2')
    plt.title('Count Plot of Response')
    plt.xlabel('Response')
    plt.ylabel('Count')
    plt.show()


def plot_feature_distributions(df):
    """
    Create count plots for categorical features.

    Parameters:
    - df (DataFrame): DataFrame containing the features 'Gender', 'Vehicle_Age', etc.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 18))
    fig.subplots_adjust(hspace=0.5)

    sns.countplot(x='Gender', data=df, color='skyblue', ax=axes[0, 0])
    axes[0, 0].set_title('Gender Distribution')

    sns.countplot(x='Vehicle_Age', data=df, color='skyblue', ax=axes[0, 1])
    axes[0, 1].set_title('Vehicle Age Distribution')

    sns.countplot(x='Vehicle_Damage', data=df, color='skyblue', ax=axes[1, 0])
    axes[1, 0].set_title('Vehicle Damage Distribution')

    sns.countplot(x='Driving_License', data=df, color='skyblue', ax=axes[1, 1])
    axes[1, 1].set_title('Driving License Distribution')

    plt.show()


def plot_histograms(df):
    """
    Create histograms for numerical features.

    Parameters:
    - df (DataFrame): DataFrame containing features 'Age', 'Vintage', 'Annual_Premium', etc.
    """
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(df['Age'], bins=50, kde=True)
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    sns.histplot(df['Vintage'], bins=50, kde=True)
    plt.title('Histogram of Vintage')
    plt.xlabel('Customer Tenure (days)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    sns.histplot(df['Annual_Premium'], bins=50, kde=True)
    plt.title('Histogram of Annual Premium')
    plt.xlabel('Annual Premium')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    sns.histplot(df['Policy_Sales_Channel'], bins=50, kde=True)
    plt.title('Histogram of Policy Sales Channel')
    plt.xlabel('Policy Sales Channel')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_boxplots(df):
    """
    Create box plots to check for outliers.

    Parameters:
    - df (DataFrame): DataFrame containing features 'Annual_Premium', 'Age', 'Vintage', and 'Response'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(
        x='Response',
        y='Annual_Premium',
        data=df,
        hue='Response',
        palette='Set2',
        dodge=False,
        ax=axes[0])
    axes[0].set_title('Annual Premium Distribution by Response')
    axes[0].legend([], [], frameon=False)

    sns.boxplot(
        x='Response',
        y='Age',
        data=df,
        hue='Response',
        palette='Set2',
        dodge=False,
        ax=axes[1])
    axes[1].set_title('Age Distribution by Response')
    axes[1].legend([], [], frameon=False)

    sns.boxplot(
        x='Response',
        y='Vintage',
        data=df,
        hue='Response',
        palette='Set2',
        dodge=False,
        ax=axes[2])
    axes[2].set_title('Vintage vs Response')
    axes[2].legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(df):
    """
    Create count plots for categorical features against the target feature.

    Parameters:
    - df (DataFrame): DataFrame containing features 'Gender', 'Vehicle_Age', 'Vehicle_Damage', and 'Response'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.countplot(
        x='Response',
        hue='Gender',
        data=df,
        palette='pastel',
        ax=axes[0])
    axes[0].set_title('Response vs Gender')
    axes[0].legend(title='Gender', loc='upper right')

    sns.countplot(
        x='Response',
        hue='Vehicle_Age',
        data=df,
        palette='Set2',
        ax=axes[1])
    axes[1].set_title('Response vs Vehicle Age')
    axes[1].legend(title='Vehicle Age', loc='upper right')

    sns.countplot(
        x='Response',
        hue='Vehicle_Damage',
        data=df,
        palette='Set3',
        ax=axes[2])
    axes[2].set_title('Response vs Vehicle Damage')
    axes[2].legend(title='Vehicle Damage', loc='upper right')

    plt.tight_layout()
    plt.show()
