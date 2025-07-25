# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.model_selection import train_test_split

#########################################################################
# DATA READING & TENSORING 

if __name__ == "__main__" :

    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import dataManager.PetScan.loader as Loader
    import dataManager.PetScan.PetScanEnlarger as Enlarger

def tensorizeScan(scan):

    # Ajouter les dimensions batch et channel : (1, 1, D, H, W)
    data = np.expand_dims(np.expand_dims(scan, axis=0), axis=0)

    # Convertir en tenseur
    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor



#########################################################################
#                               MODEL                                   #
#########################################################################

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


if __name__ == "__main__":

    #device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    device = "cuda"
    print(device)

    model = Simple3DCNN(num_classes=3).to(device)
    print(model)

    loader = Loader.PETScanLoader("../../Desktop/Cancer_pain_data/PETdata/data/", "zscore")
    scan = loader.load_scan("PHC64_4532.nii")
    tensor = tensorizeScan(scan)
    tensor = tensor.to("cuda")

    with torch.no_grad():
        output = model(tensor)
    
    print("Output:", output)
    print("Classe prédite :", torch.argmax(output, dim=1).item())

    print("===============================================================")

    # Load labelized data
    labelisedData = loader.load_all_labelised()

    #Enlarge the Dataset with geometrical modifications
    enlargementMethod = ['flip_x', 'flip_y', 'flip_z', 'noise', 'adjust_contrast', 'blur']
    EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 3)

    # Disassociate the label from the example
    X, Y = make_batch(EnlargedData)
    print("Shape X before train_test_split:", X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    print('shape de X_train :', X_train.shape)

    history = train_model(
    model,
    torch.tensor(X_train).float(),
    torch.tensor(y_train).float(),
    torch.tensor(X_val).float(),
    torch.tensor(y_val).float(),
    batch_size=30,
    epochs=50,
    criterion = nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.005),
    metric=confident_accuracy,
    device=device
)
    
    # Premier graphique
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png") 

    # Deuxième graphique
    plt.figure()
    plt.plot(history['score'], label='Train Accuracy')
    plt.plot(history['val_score'], label='Val Accuracy')
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")  

    #from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#
    #y_pred = []
    #y_true = []
#
    #model.eval()
    #with torch.no_grad():
    #    for x_batch, y_batch in DataLoader(TensorDataset(X_val, y_val), batch_size=6):
    #        preds = model(x_batch.to(device))
    #        pred_classes = preds.argmax(dim=1).cpu()
    #        y_pred.extend(pred_classes)
    #        y_true.extend(y_batch)
#
    #cm = confusion_matrix(y_true, y_pred)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    #plt.show()
