import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
from models.model.naiveModel import Simple3DCNN

# Training & metics functions
import models.communs.trainingfunction as training_fn
import torch.optim as optim
import models.communs.metrics as metric

# Batch creator
from dataManager.PetScan.batch import batch

# Overviewing the result 
import models.communs.performanceAnalyser as plotter

if __name__ == "__main__":
    
    #device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    print("device :",device)

    model = Simple3DCNN(num_classes=2).to(device)
    print("model structure looks like :",model)

    print(" Loading and creating the batch:")
    batch = batch(data_dir=os.path.expanduser("~/Documents/CancerPain/PETdata/data"), 
                  preprocessing_method=['mean_template'],
                  normalization='zscore')
    X_train, X_val, y_train, y_val = batch.split_train_test(['blur', 'noise'])


    print(" Starting training process :")
    history = training_fn.train_model(
    model,
    torch.tensor(X_train).float(),
    torch.tensor(y_train).float(),
    torch.tensor(X_val).float(),
    torch.tensor(y_val).float(),
    batch_size=4,
    epochs=5,
    criterion = nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.005),
    metric=metric.confident_accuracy,
    device=device)

    plotter.plot_confusion_matrix(model, X_val, y_val, class_names=['class 0', 'class 1'])
    plotter.compute_sensitivity(model, X_val, y_val)
    plotter.compute_accuracy(model, X_val, y_val)
    plotter.plot_prediction_confidence_gap(model, X_val, y_val)
    plotter.plot_roc_curves(model, X_val, y_val, num_classes=2)
    plotter.classification_report_summary(model, X_val, y_val, class_names=['class 0', 'class 1'])