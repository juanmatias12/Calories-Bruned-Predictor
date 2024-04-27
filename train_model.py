from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from load_data import load_datasets
from process_data import preprocess_data
import joblib


def train_and_evaluate_models(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBR': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGB': XGBRegressor(n_estimators=100, random_state=42)
    }

    # Dictionary to store model performance
    model_performance = {}

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train the model
        predictions = model.predict(X_test)  # Make predictions
        mae = mean_absolute_error(y_test, predictions)  # Evaluate the model
        model_performance[name] = {'MAE': mae, 'predictions': predictions, 'y_test': y_test}
        print(f'{name} MAE: {mae}')


        #Saving the XGB model to a file after training

        if name == 'XGB':
            joblib.dump(model, 'xgb_model.pkl')
            print(f'{name} Model saved to xgb_model.pkl')

    return model_performance



if __name__ == '__main__':
    # Define the path to the datasets
    calories_path = '/Users/chico/Documents/CS450 Calories Burned Final/calories.csv'
    exercise_path = '/Users/chico/Documents/CS450 Calories Burned Final/exercise.csv'

    # Load and preprocess data
    combined_data = load_datasets(calories_path, exercise_path)
    processed_data = preprocess_data(combined_data)

    # Define the target column name (update this to your actual target column)
    target_column = 'Calories'

    # Split the processed data into features and target
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and evaluate performance
    performance = train_and_evaluate_models(X, y)


