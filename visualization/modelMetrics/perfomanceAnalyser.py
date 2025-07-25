import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import label_binarize


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


def get_model_predictions(model, X):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X).float().to(next(model.parameters()).device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


def plot_confusion_matrix(model, X_test, y_test, class_names=None):
    preds, _ = get_model_predictions(model, X_test)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def compute_sensitivity(model, X_test, y_test):
    preds, _ = get_model_predictions(model, X_test)
    cm = confusion_matrix(y_test, preds)
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    sensitivity = TP / (TP + FN + 1e-6)
    return sensitivity  # array per class


def compute_accuracy(model, X_test, y_test):
    preds, _ = get_model_predictions(model, X_test)
    return accuracy_score(y_test, preds)


def plot_prediction_confidence_gap(model, X_test, y_test):
    _, probs = get_model_predictions(model, X_test)
    ideal = np.zeros_like(probs)
    ideal[np.arange(len(y_test)), y_test] = 1.0
    diffs = np.abs(probs - ideal)
    scores = np.sum(diffs, axis=1)

    plt.hist(scores, bins=20, color='purple', edgecolor='black')
    plt.xlabel("Total confidence error (ideal = 0)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of deviation from perfect prediction")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(model, X_test, y_test, num_classes=None):
    _, probs = get_model_predictions(model, X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(probs.shape[1])))

    if num_classes is None:
        num_classes = probs.shape[1]

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_test_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def classification_report_summary(model, X_test, y_test, class_names=None):
    preds, _ = get_model_predictions(model, X_test)
    print(classification_report(y_test, preds, target_names=class_names))
