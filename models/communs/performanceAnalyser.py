# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

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
        fpr, tpr, _ = roc_curve(y_test_bin[:], probs[:, i])
        auc = roc_auc_score(y_test_bin[:], probs[:, i])
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



from sklearn.model_selection import StratifiedKFold

def cross_validate_model(X, y, model_class, model_args={}, 
                         train_fn=None, metric_fn=None, 
                         epochs=10, batch_size=4, n_splits=5, seed=42, device='cpu'):
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create a fresh model for each fold
        model = model_class(**model_args).to(device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()

        # Train
        history = train_fn(
            model,
            torch.tensor(X_train).float(), 
            torch.tensor(y_train).long(),
            torch.tensor(X_val).float(), 
            torch.tensor(y_val).long(),
            criterion,
            optimizer,
            metric=metric_fn,
            epochs=epochs,
            batch_size=batch_size,
            device=device
        )

        # Evaluate
        #final_val_score = history['val_score'][-1]
        #fold_metrics.append(final_val_score)
        #print(f"[Fold {fold+1}] Validation Score: {final_val_score:.4f}")
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val).float().to(device)
            y_val_tensor = torch.tensor(y_val).long().to(device)

            preds = model(X_val_tensor)
            val_score = metric_fn(preds, y_val_tensor) if metric_fn is not None else 0

        fold_metrics.append(val_score)
        print(f"[Fold {fold+1}] Validation Score: {val_score:.4f}")


    avg = np.mean(fold_metrics)
    std = np.std(fold_metrics)
    print(f"\n Cross-Validation Result: {avg:.4f} ± {std:.4f}")
    return fold_metrics