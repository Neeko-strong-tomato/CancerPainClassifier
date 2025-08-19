import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_features(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].values             
    X = df.drop(columns=["label"]).values
    return X, y


def stratified_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def make_dataloaders(X_train, y_train, X_test, y_test, batch_size=16):
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    X, y = load_features("metricExtraction/pet_features.csv")
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

    train_loader, test_loader = make_dataloaders(X_train, y_train, X_test, y_test)

    for xb, yb in train_loader:
        print("Batch X:", xb.shape)
        print("Batch y:", yb.shape)
        break
