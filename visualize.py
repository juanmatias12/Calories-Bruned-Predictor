import matplotlib.pyplot as plt


def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color= 'red', label = 'Predictions')
    plt.scatter(y_test, y_test, alpha=0.5, color='blue', label = 'Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')  # Identity line
    plt.show()
