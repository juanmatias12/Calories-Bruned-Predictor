import pandas as pd


def load_datasets(calories_path, exercise_path):
    # Load datasets
    calories = pd.read_csv('/Users/chico/Documents/CS450 Calories Burned Final/calories.csv')
    exercise_data = pd.read_csv('/Users/chico/Documents/CS450 Calories Burned Final/exercise.csv')

    # Assuming both datasets contain a common column 'UserID' to join on
    combined_data = pd.concat([exercise_data, calories['Calories']], axis=1)

    return combined_data


if __name__ == "__main__":
    calories_path = '/Users/chico/Documents/CS450 Calories Burned Final/calories.csv'
    exercise_path = '/Users/chico/Documents/CS450 Calories Burned Final/exercise.csv'
    dataset = load_datasets(calories_path, exercise_path)
    print(dataset.head())
