import pandas as pd


def load_dataset(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    print(data.columns)
    return data


if __name__ == "__main__":
    file_path = '/Users/chico/Documents/CS450 Calories Burned Final/exercise_dataset.csv'
    dataset = load_dataset(file_path)
    print(dataset.head())
