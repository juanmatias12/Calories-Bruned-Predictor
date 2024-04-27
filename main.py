from load_data import load_datasets
from process_data import preprocess_data
from train_model import train_and_evaluate_models
from visualize import plot_predictions, plot_performance

def main():
    # Paths to your datasets
    calories_path = '/Users/chico/Documents/CS450 Calories Burned Final/calories.csv'
    exercise_path = '/Users/chico/Documents/CS450 Calories Burned Final/exercise.csv'

    # Load and preprocess the data
    print("Loading and preprocessing data...")
    combined_data = load_datasets(calories_path, exercise_path)
    processed_data = preprocess_data(combined_data)

    # Assuming 'Calories' is your target column, adjust if it's different
    target_column = 'Calories'
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]

    # Train models and evaluate performance
    print("Training and evaluating models...")
    performance = train_and_evaluate_models(X, y)

    # Visualize the performance of models
    print("Visualizing model performance...")
    plot_performance(performance)

    # Visualize predictions for each model
    for model_name, info in performance.items():
        print(f"Visualizing predictions for {model_name}...")
        plot_predictions(actual=info['y_test'], predicted=info['predictions'], model_name=model_name)


if __name__ == "__main__":
    main()
