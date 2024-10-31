import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE


def cap_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
    """
    Cap outliers in a specified column using quantile-based thresholds.
    """
    lower_threshold = df[column].quantile(lower_quantile)
    upper_threshold = df[column].quantile(upper_quantile)
    df[column] = df[column].clip(lower=lower_threshold, upper=upper_threshold)
    return df


def label_vehicle_category(row):
    """
    Label the vehicle category based on Vehicle_Age and Vehicle_Damage.
    """
    if row['Vehicle_Age'] == '< 1 Year':
        return 'New-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'New-Undamaged'
    elif row['Vehicle_Age'] == '> 2 Years':
        return 'Old-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Old-Undamaged'
    else:
        return 'Mid-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Mid-Undamaged'


def create_age_vehicle_damage_category(row):
    """
    Create an age and vehicle damage combined category feature.
    """
    age_group = 'Young' if row['Age'] < 30 else 'Middle-Aged' if row['Age'] < 50 else 'Senior'
    return f"{age_group} - {
        'Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Undamaged'}"


def create_vehicle_age_insurance_risk(row):
    """
    Create an insurance risk level based on vehicle age and previous insurance status.
    """
    if row['Vehicle_Age'] == '< 1 Year':
        return 'Low Risk' if row['Previously_Insured'] == 'Yes' else 'Medium Risk'
    elif row['Vehicle_Age'] == '1-2 Years':
        return 'Medium Risk'
    else:
        return 'High Risk' if row['Previously_Insured'] == 'No' else 'Medium Risk'


def create_premium_risk_category(row):
    """
    Create a risk level based on vehicle damage and annual premium.
    """
    if row['Vehicle_Damage'] == 'Yes':
        return 'High Risk' if row['Annual_Premium'] > 2000 else 'Medium Risk'
    else:
        return 'Low Risk'


def engineer_features(df):
    """
    Apply feature engineering on the dataset.
    """
    # Cap outliers in 'Annual_Premium'
    df = cap_outliers(df, 'Annual_Premium')

    # Create new features
    df['Vehicle_Category'] = df.apply(label_vehicle_category, axis=1)
    df['Age_Vehicle_Damage'] = df.apply(
        create_age_vehicle_damage_category, axis=1)
    df['Vehicle_Age_Risk'] = df.apply(
        create_vehicle_age_insurance_risk, axis=1)
    df['Premium_Risk'] = df.apply(create_premium_risk_category, axis=1)

    return df


def split_data(df):
    """
    Split data into training and test sets with separate features and target.
    """
    df_train = df[df['Response'] != -1]
    df_test = df[df['Response'] == -1].drop(columns=['Response'])

    X = df_train.drop(columns=["Response"], axis=1)
    y = df_train["Response"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, df_test


def preprocess_data(X_train, X_test, df_test):
    """
    Preprocess data using standard scaling and one-hot encoding.
    """
    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=['object', 'category']).columns.tolist()

    # Set up column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',
             StandardScaler(),
             numerical_features),
            ('cat',
             OneHotEncoder(
                 drop='first',
                 handle_unknown='ignore'),
                categorical_features)])

    # Transform datasets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    df_test_processed = preprocessor.transform(df_test)

    return X_train_processed, X_test_processed, df_test_processed, preprocessor


def balance_data(X_train, y_train):
    """
    Balance the training data using SMOTE.
    """
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced
