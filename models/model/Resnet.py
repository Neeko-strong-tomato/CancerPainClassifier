# CancerPainClassifier
# Copyright (c) 2025 Neeko
# License: MIT
# If used in research, please cite: https://github.com/Neeko-strong-tomato/CancerPainClassifier

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

if __name__ == '__main__' :

    import torch.optim as optim
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
    enlargementMethod = ['adjust_contrast', 'blur', 'noise', 'adjust_brightness']
    #EnlargedData = Enlarger.augmentate_batch(labelisedData, enlargementMethod, True, 2)

    # Disassociate the label from the example
    X, Y = make_batch(labelisedData)
    print("Shape X before train_test_split:", X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, y_train = Enlarger.augmentate_dataset_separated(X_train, y_train, enlargementMethod, True, 2)
    X_val, y_val = Enlarger.augmentate_dataset_separated(X_val, y_val, enlargementMethod, True, 2)

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


    #print(model)
    print(np.shape(X_train))

    history = train_mednet_model(
    model,
    torch.tensor(X_train).float(),
    torch.tensor(y_train).float(),
    torch.tensor(X_val).float(),
    torch.tensor(y_val).float(),
    batch_size=22, #22
    epochs=35, #35
    criterion = nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.00007),
    metric=accuracy_metric,
    device=device,
    freeze_up_to=4
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


    import visualization.modelMetrics.perfomanceAnalyser as plotter

    plotter.plot_confusion_matrix(model, X_val, y_val, class_names=['class 0', 'class 1', 'class 2'])
    plotter.compute_sensitivity(model, X_val, y_val)
    plotter.compute_accuracy(model, X_val, y_val)
    plotter.plot_prediction_confidence_gap(model, X_val, y_val)
    plotter.plot_roc_curves(model, X_val, y_val, num_classes=3)
    plotter.classification_report_summary(model, X_val, y_val, class_names=['class 0', 'class 1', 'class 2'])