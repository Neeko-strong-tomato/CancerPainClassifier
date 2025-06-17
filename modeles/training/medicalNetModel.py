import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from typing import Union
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from collections import OrderedDict

# Téléchargement et chargement du backbone de MedicalNet
def generate_resnet18_model(in_channels=1):
    from medicalnet.models import resnet

    model = resnet.generate_model(model_depth=18,
                                  n_input_channels=in_channels,
                                  shortcut_type='B',
                                  num_classes=400)  # Dummy output classes, we'll change the head

    # Charger les poids préentraînés
    weights = torch.load("pretrained/resnet_18_23dataset.pth", map_location='cuda')
    new_state_dict = OrderedDict()
    for k, v in weights['state_dict'].items():
        name = k.replace("module.", "")  # supprimer "module." pour compatibilité
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model

class MedicalNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, pretrained=True,
                 head_layers='default', device='cpu',
                 weights_path="~/Downloads/MedicalNet_pytorch_files2/pretrain/resnet_18_23dataset.pth"):
        """
        Modèle MedicalNet (ResNet18 3D) pour la classification de PET scans.

        Args:
            in_channels: Canaux d'entrée (1 pour PET).
            num_classes: Nombre de classes de sortie.
            pretrained: Charger les poids MedicalNet.
            head_layers: 'replace' ou 'extend' la tête.
            device: 'cpu' ou 'cuda'.
            weights_path: Chemin vers les poids MedicalNet.
        """
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Charger ResNet18 MedicalNet
        self.model = self._load_backbone(pretrained, weights_path)

        # Modifier la tête
        if head_layers == 'replace':
            self._replace_head()
        elif head_layers == 'extend':
            self._extend_head()

        self.to(device)

    def _load_backbone(self, pretrained, weights_path):
        
        import modeles.storage.medicalnetModel as medicalnetModel
        
        model = medicalnetModel.generate_model(
            model_depth=18,
            n_input_channels=self.in_channels,
            shortcut_type='B',
            num_classes=400  # Dummy: remplacé ensuite
        )
        if pretrained:
            print("[INFO] Chargement des poids MedicalNet...")
            weights = torch.load(weights_path, map_location='cuda')
            new_state_dict = OrderedDict()
            for k, v in weights['state_dict'].items():
                new_state_dict[k.replace("module.", "")] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("[INFO] Poids chargés.")
        return model

    def _replace_head(self):
        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
    )


    def _extend_head(self):
        old_head = self.model.conv_seg
        self.model.conv_seg = nn.Sequential( old_head,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(400, 256),  # ou adapte selon la sortie du old_head
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
    )


    def forward(self, x):
        return self.model(x)

    def predict(self, scan: Union[torch.Tensor, list, np.ndarray]):
        self.eval()
        with torch.no_grad():
            if isinstance(scan, (list, np.ndarray)):
                scan = torch.tensor(scan, dtype=torch.float32)
            if scan.dim() == 4:
                scan = scan.unsqueeze(0)
            scan = scan.to(self.device)
            output = self.forward(scan)
            pred = torch.argmax(output, dim=1)
            return pred().numpy()



def freeze_model_layers(model: MedicalNetClassifier, num_layers_to_freeze: int = 2):
    """
    Freeze the first `num_layers_to_freeze` residual blocks in MedicalNet.
    `model` is a MedicalNetClassifier with `model.model` = ResNet.
    """
    resnet = model.model  # 🔄 raccourci

    # Liste des couches résiduelles à freezer
    layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]

    for i, layer in enumerate(layers):
        requires_grad = i >= num_layers_to_freeze
        for param in layer.parameters():
            param.requires_grad = requires_grad

    # Toujours laisser les dernières couches actives
    for param in resnet.conv_seg.parameters():
        param.requires_grad = True

    # (Optionnel) laisser les premières couches dégelées ?
    for param in resnet.conv1.parameters():
        param.requires_grad = True
    for param in resnet.bn1.parameters():
        param.requires_grad = True

    # Débogage
    #print("✅ Frozen layers configuration:")
    #for name, param in model.named_parameters():
    #    print(f" - {name}: requires_grad={param.requires_grad}")




def train_mednet_model(model, 
                      X_train, y_train, X_val, y_val,
                      criterion=None, optimizer=None, metric=None,
                      batch_size=4, epochs=20,
                      device='cpu',
                      freeze_up_to=0):
    """
    Entraîne un modèle MONAI (avec option de figer les premières couches).

    Args:
        freeze_up_to: int, nombre de couches à figer (défaut = 0 = rien)
    """

    model.to(device)

    # Fige les couches souhaitées
    if freeze_up_to > 0:
        freeze_model_layers(model, freeze_up_to)

    X_train, y_train = X_train.to(device), y_train.to(device).long()
    X_val, y_val = X_val.to(device), y_val.to(device).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    history = {'loss': [], 'val_loss': [], 'score': [], 'val_score': []}

    for epoch in range(epochs):
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

        # Validation
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

        # Logging
        train_loss /= len(train_loader.dataset)
        train_score /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_score /= len(val_loader.dataset)

        history['loss'].append(train_loss)
        history['score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)

        print(f"Epoch {epoch+1:02}/{epochs} | "
              f"Loss: {train_loss:.4f} | Score: {train_score:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f}")

    return history


def train_on_full_data(model, X, Y, 
                      criterion=None, optimizer=None, metric=None,
                      batch_size=4, epochs=20,
                      device='cpu',
                      freeze_up_to=0):

    epochs = len(X)//batch_size

    model.to(device)

    # Fige les couches souhaitées
    if freeze_up_to > 0:
        freeze_model_layers(model, freeze_up_to)

    train_loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=False)

    history = {'loss': [], 'val_loss': [], 'score': [], 'val_score': []}

    for epoch in range(epochs):
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

        train_loss /= len(train_loader.dataset)
        train_score /= len(train_loader.dataset)

        history['loss'].append(train_loss)
        history['score'].append(train_score)

        print(f"Epoch {epoch+1:02}/{epochs} | "
              f"Loss: {train_loss:.4f} | Score: {train_score:.4f} |")
    
    return history


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


def confident_accuracy(outputs, targets):
    """
    Calcule l'accuracy pondérée par la confiance dans la prédiction correcte.
    Plus le modèle est confiant et correct, mieux c’est.
    """
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    correct = preds == targets

    # score = probabilité associée à la bonne classe (si correcte)
    confidence = probs[range(len(targets)), targets]

    # Ne garde que les bonnes prédictions, pondérées par leur confiance
    confident_correct = confidence[correct]
    
    return confident_correct.mean().item()  # moyenne de la confiance sur les bonnes prédictions

def accuracy_metric(preds, labels):
    preds_class = preds.argmax(dim=1)   
    correct = (preds_class == labels).sum().item()
    return correct / labels.size(0)

if __name__ == '__main__' :

    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    import dataLoaders.PETScanLoader as Loader
    import dataLoaders.PetScanEnlarger as Enlarger


    def make_batch(data):
        X = []
        Y = []

        for patient in data:
            X.append(patient["data"])
            Y.append(patient["label"])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int64)


        if len(X.shape) == 4:
            X = np.expand_dims(X, axis=1)  # (N, D, H, W) -> (N, 1, D, H, W)

        return X, Y

    print("===============================================================")

    device = "cuda"
    print(device)

    model = MedicalNetClassifier(
        in_channels=1,
        num_classes=3,
        pretrained=True,
        head_layers='extend',
        device=device,
        weights_path="modeles/storage/resnet_18_23dataset.pth"
    )
    model.to(device)


    # Load labelized data
    loader = Loader.PETScanLoader("../../Desktop/Cancer_pain_data/PETdata/data/", "zscore")
    labelisedData = loader.load_all_labelised()

    #Enlarge the Dataset with geometrical modifications
    enlargementMethod = ['flip_x', 'flip_y', 'noise', 'flip_z']
    EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 3)

    # Disassociate the label from the example
    X, Y = make_batch(EnlargedData)
    print("Shape X before train_test_split:", X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    #cross_validate_model(X, Y, MedicalNetClassifier, 
    #                     model_args={ 'in_channels':1,
    #                                  'num_classes':3, 
    #                                  'pretrained':True, 
    #                                  'head_layers':'extend', 
    #                                  'device':device, 
    #                                  'weights_path':"modeles/storage/resnet_18_23dataset.pth"}, 
    #                    train_fn=train_mednet_model, metric_fn=accuracy_metric, 
    #                    epochs=35, batch_size=12, 
    #                    n_splits=5, seed=42, device=device)


    print(model)

    history = train_mednet_model(
    model,
    torch.tensor(X_train).float(),
    torch.tensor(y_train).float(),
    torch.tensor(X_val).float(),
    torch.tensor(y_val).float(),
    batch_size=15,
    epochs=24,
    criterion = nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    metric=accuracy_metric,
    device=device,
    freeze_up_to=2
    )


    
    plt.figure(1)
    plt.title("Mean Absolute score")
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


    import visualization.modelMetrics.trainingData as plotter

    plotter.plot_confusion_matrix(model, X_val, y_val, class_names=['class 0', 'class 1', 'class 2'])
    plotter.compute_sensitivity(model, X_val, y_val)
    plotter.compute_accuracy(model, X_val, y_val)
    plotter.plot_prediction_confidence_gap(model, X_val, y_val)
    plotter.plot_roc_curves(model, X_val, y_val, num_classes=3)
    plotter.classification_report_summary(model, X_val, y_val, class_names=['class 0', 'class 1', 'class 2'])