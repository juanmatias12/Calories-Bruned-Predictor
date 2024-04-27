import matplotlib.pyplot as plt
import seaborn as sns


def plot_predictions(actual, predicted, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.6, color='blue', label='Predicted')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=2, label='Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values for {model_name}')
    plt.legend()
    plt.show()


def plot_loss(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_performance(performance):
    models = list(performance.keys())
    maes = [info['MAE'] for info in performance.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(models, maes, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of Model Performance')
    plt.show()
