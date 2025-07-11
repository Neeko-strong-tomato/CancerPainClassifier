import torch

# Metrics 

def accuracy_metric(preds, labels):
    preds_class = preds.argmax(dim=1)   
    correct = (preds_class == labels).sum().item()
    return correct / labels.size(0)

def confident_accuracy(preds, labels, threshold=0.8):
    probs = torch.softmax(preds, dim=1)
    confidences, pred_classes = probs.max(dim=1)
    
    correct = (pred_classes == labels)
    confident = (confidences >= threshold)

    # Compte uniquement les prédictions correctes et confiantes
    selected = correct & confident
    return selected.float().sum().item() / len(labels)


def certainty_gap(preds, labels):
    probs = torch.softmax(preds, dim=1)
    target_probs = probs[range(len(labels)), labels]
    return (1 - target_probs).mean()  