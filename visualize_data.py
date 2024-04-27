import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_datasets
from process_data import preprocess_data


def plot_histogram(data, column, color='skyblue'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], color=color, kde=True)
    plt.title(f'Distribution of {column} - Explains the frequency of different values in {column}')
    plt.xlabel(f'{column} Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_category_counts(data, column, palette='lightgreen'):
    plt.figure(figsize=(10, 6))
    sns.countplot(data[column], palette=palette)
    plt.title(f'Frequency of Different {column} Types')
    plt.xlabel(f'{column} Type')
    plt.xticks(rotation=45)
    plt.ylabel('Counts')
    plt.show()


def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=correlation_matrix.columns,
                yticklabels=correlation_matrix.columns)
    plt.title('Correlation Matrix - Shows the relationship strength between variables')
    plt.show()


# Assuming this file is being run as a script
if __name__ == '__main__':
    # Load data
    data = load_datasets('/Users/chico/Documents/CS450 Calories Burned Final/calories.csv', '/Users/chico/Documents/CS450 Calories Burned Final/exercise.csv')
    data = preprocess_data(data)
    # Plotting data distributions for Calories if present
    if 'Calories' in data.columns:
        plot_histogram(data, 'Calories')

    # If there's a categorical feature like 'Activity'
    if 'Activity' in data.columns:
        plot_category_counts(data, 'Activity')

    # If you have 'Heart_Rate' and want to visualize it
    if 'Heart_Rate' in data.columns:
        plot_histogram(data, 'Heart_Rate', color='red')

    # Plot correlation matrix for numerical features
    plot_correlation_matrix(data)

