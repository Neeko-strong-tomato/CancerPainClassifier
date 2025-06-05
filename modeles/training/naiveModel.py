import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
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

    import dataLoaders.PETScanLoader as Loader
    import dataLoaders.PetScanEnlarger as Enlarger

def tensorizeScan(scan):

    # Ajouter les dimensions batch et channel : (1, 1, D, H, W)
    data = np.expand_dims(np.expand_dims(scan, axis=0), axis=0)

    # Convertir en tenseur
    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor
#########################################################################

# Data Formating

def make_batch(data):
    X = []
    Y = []

    for patient in data:
        X.append(patient["data"])
        Y.append(patient["label"])

    for i, scan in enumerate(X):
            if scan.shape != X[0].shape:
                print(f"Inconsistent shape at index {i}: {scan.shape} vs {X[0].shape}")
                fix_shape(scan)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)


    if len(X.shape) == 4:
        X = np.expand_dims(X, axis=1)  # (N, D, H, W) -> (N, 1, D, H, W)

    return X, Y



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



def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, 
                criterion=None, optimizer=None, metric=None, device='cpu'):
    
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device).long()
    X_val, y_val = X_val.to(device), y_val.to(device).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    history = {'loss': [], 'val_loss': [], 'score': [], 'val_score': []}

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_score = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).long()

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            score = metric(preds, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_score += score * x_batch.size(0)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_score = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).long()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                score = metric(preds, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                val_score += score * x_batch.size(0)

        # ---------- Logging ----------
        train_loss /= len(train_loader.dataset)
        train_score /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_score /= len(val_loader.dataset)

        history['loss'].append(train_loss)
        history['score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)

        print(f"Epoch {epoch+1:03}/{epochs} | "
              f"Loss: {train_loss:.4f} | Score: {train_score:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}")

    return history


def accuracy_metric(preds, labels):
    preds_class = preds.argmax(dim=1)   
    correct = (preds_class == labels).sum().item()
    return correct / labels.size(0)


if __name__ == "__main__":

    #device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    print(device)

    model = Simple3DCNN(num_classes=3).to(device)
    print(model)

    loader = Loader.PETScanLoader("/Volumes/UBUNTU 22_0/PETdata/data/", "zscore")
    scan = loader.load_scan("PHC64_4532.nii")
    tensor = tensorizeScan(scan)

    with torch.no_grad():
        output = model(tensor)
    
    print("Output:", output)
    print("Classe prédite :", torch.argmax(output, dim=1).item())

    print("===============================================================")

    # Load labelized data
    labelisedData = loader.load_all_labelised()

    #Enlarge the Dataset with geometrical modifications
    enlargementMethod = ['flip_x', 'flip_y', 'noise']
    EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 3)

    # Disassociate the label from the example
    X, Y = make_batch(EnlargedData)
    print("Shape X before train_test_split:", X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    history = train_model(
    model,
    torch.tensor(X_train).float(),
    torch.tensor(y_train).float(),
    torch.tensor(X_val).float(),
    torch.tensor(y_val).float(),
    batch_size=10,
    epochs=30,
    criterion = nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.005),
    metric=accuracy_metric
)
    
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

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in DataLoader(TensorDataset(X_val, y_val), batch_size=6):
            preds = model(x_batch.to(device))
            pred_classes = preds.argmax(dim=1).cpu()
            y_pred.extend(pred_classes)
            y_true.extend(y_batch)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
