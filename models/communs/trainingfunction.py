import torch
from torch.utils.data import TensorDataset, DataLoader

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