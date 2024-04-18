import sys
from load_data import load_dataset
from process_data import preprocess_data
from features import split_data
from train_model import train_linear_regression, train_random_forest, train_gbm
from evaluate_model import evaluate_model
from visualize import plot_predictions

def main():
    file_path = '/Users/chico/Documents/CS450 Calories Burned Final/exercise_dataset.csv'
    target_column = 'Calories per kg'

    # Load and preprocess the data
    data = load_dataset(file_path)
    processed_data = preprocess_data(data)

    # Prepare the data
    X = processed_data.drop(columns=['Activity, Exercise or Sport (1 hour)', target_column], errors='ignore')
    y = processed_data[target_column]

    # Assuming command-line argument to choose the model type
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'logistic_regression'

    if model_type == 'random_forest':
        model = train_random_forest(X, y)
    elif model_type == 'gbm':
        model = train_gbm(X, y)
    elif model_type == 'linear_regression':
        model = train_linear_regression(X,y)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Evaluate the chosen model
    predictions = model.predict(X)
    mae = evaluate_model(y, predictions)
    print(f'{model_type.upper()} MAE: {mae}')

    #Plotting the predictions against the actual values
    plot_predictions(y, predictions)

if __name__ == '__main__':
    main()
