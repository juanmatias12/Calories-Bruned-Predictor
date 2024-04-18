from sklearn.model_selection import train_test_split


def split_data(X,y):

    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_holdout_set(X, y):
    # Split the data into a holdout set and the rest
    X, X_holdout, y, y_holdout = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    return X, X_holdout, y, y_holdout
