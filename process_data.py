import pandas as pd


def preprocess_data(data):
    # Check if the 'Gender' column exists and convert it. Can't use string values. 
    if 'Gender' in data.columns:
        # Map 'male' to 0 and 'female' to 1
        gender_map = {'male': 0, 'female': 1}
        data['Gender'] = data['Gender'].map(gender_map)

    # Convert categorical variables to numeric using one-hot encoding
    # Assuming other categorical columns need to be transformed if they exist
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Convert all columns to numeric, coercing errors for any remaining non-numeric columns
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill missing values, if any, using forward fill
    data.ffill(inplace=True)

    return data
