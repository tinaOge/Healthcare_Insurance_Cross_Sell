import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def cap_outliers(df):
    """Cap outliers in the Annual_Premium column."""
    lower_threshold = df['Annual_Premium'].quantile(0.05)
    upper_threshold = df['Annual_Premium'].quantile(0.95)
    df['Annual_Premium'] = df['Annual_Premium'].clip(lower=lower_threshold, upper=upper_threshold)


def label_vehicle_category(row):
    """Label vehicle category based on Vehicle_Age and Vehicle_Damage."""
    if row['Vehicle_Age'] == '< 1 Year':
        return 'New-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'New-Undamaged'
    elif row['Vehicle_Age'] == '> 2 Years':
        return 'Old-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Old-Undamaged'
    else:
        return 'Mid-Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Mid-Undamaged'


def create_age_vehicle_damage_category(row):
    """Create a category based on Age and Vehicle_Damage."""
    age_group = 'Young' if row['Age'] < 30 else 'Middle-Aged' if row['Age'] < 50 else 'Senior'
    return f"{age_group} - {'Damaged' if row['Vehicle_Damage'] == 'Yes' else 'Undamaged'}"


def create_vehicle_age_insurance_risk(row):
    """Assign insurance risk based on Vehicle_Age and Previously_Insured."""
    if row['Vehicle_Age'] == '< 1 Year':
        return 'Low Risk' if row['Previously_Insured'] == 'Yes' else 'Medium Risk'
    elif row['Vehicle_Age'] == '1-2 Years':
        return 'Medium Risk'
    else:  # '> 2 Years'
        return 'High Risk' if row['Previously_Insured'] == 'No' else 'Medium Risk'


def create_premium_risk_category(row):
    """Create premium risk category based on Vehicle_Damage and Annual_Premium."""
    if row['Vehicle_Damage'] == 'Yes':
        return 'High Risk' if row['Annual_Premium'] > 2000 else 'Medium Risk'
    else:
        return 'Low Risk'


def preprocess_data(df):
    """Preprocess the DataFrame by capping outliers and creating new features."""
    cap_outliers(df)

    # Create new features
    df['Vehicle_Category'] = df.apply(label_vehicle_category, axis=1)
    df['Age_Vehicle_Damage'] = df.apply(create_age_vehicle_damage_category, axis=1)
    df['Vehicle_Age_Risk'] = df.apply(create_vehicle_age_insurance_risk, axis=1)
    df['Premium_Risk'] = df.apply(create_premium_risk_category, axis=1)

    # One-Hot Encoding
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

    # Scale numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features.remove('Response')  # Exclude the target variable
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_features])
    scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_features)

    # Concatenate all features
    final_df = pd.concat(
        [scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True), df['Response'].reset_index(drop=True)],
        axis=1)

    # Feature selection
    X = final_df.drop(columns=['Response'])
    y = final_df['Response']
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    selected_columns = X.columns[selector.get_support()]

    # Retain only selected columns
    df_cluster = final_df[selected_columns]

    return df_cluster, y,  selected_columns  # Returning the features and target variable
