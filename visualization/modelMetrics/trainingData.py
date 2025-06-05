import matplotlib.pyplot as plt


def plot_training_data(history):
    """
    Plots the training and validation metrics from the training history.

    Parameters:
    history (dict): A dictionary containing training history with keys 'loss', 'val_loss', 'score', 'val_score'.
    """
    
    if not isinstance(history, dict):
        raise ValueError("History must be a dictionary containing training metrics.")

    # Check if required keys are in the history
    required_keys = {'loss', 'val_loss', 'score', 'val_score'}
    if not required_keys.issubset(history.keys()):
        raise KeyError(f"History must contain the following keys: {required_keys}")

    plt.figure(1)
    plt.title("Mean Absolute Error")
    plt.xlabel("#Epoch")
    plt.plot(history['score'], label='Training Score')
    plt.plot(history['val_score'], label='Validation Score')
    plt.legend()

    plt.figure(2)
    plt.title("Loss")
    plt.xlabel("#Epoch")
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.show()