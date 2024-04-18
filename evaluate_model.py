from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    return mae


def perform_cross_validation(X, y):
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')
    mae_scores = -scores
    print(f"Cross-validated MAEs: {mae_scores}")
    print(f"Mean MAE: {mae_scores.mean()}")
    print(f"Standard Deviation of MAE: {mae_scores.std()}")
