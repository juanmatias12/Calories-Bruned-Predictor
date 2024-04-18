import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data):
    # Convert all columns except the first to numeric
    # The first column 'Activity, Exercise or Sport (1 hour)' should remain as text
    cols = data.columns.drop('Activity, Exercise or Sport (1 hour)')
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

    # Now, only drop rows where the numeric conversions failed (if any)
    data.dropna(subset=cols, inplace=True)

    return data



def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled