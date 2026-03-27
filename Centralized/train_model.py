import os
import glob
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRAIN] Using device: {device}")

# ── Same architecture as federated clients ──────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.net.to(device)

    def forward(self, x):
        return self.net(x)


class HospitalDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── Main training function ───────────────────────────────────────────────────
def train_centralized_model(data_dir: str) -> float:
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"[TRAIN] Merging {len(csv_files)} hospital dataset(s): {[os.path.basename(f) for f in csv_files]}")

    frames = [pd.read_csv(f, header=None) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    X = df.iloc[:, :21].values.astype("float32")
    y = df.iloc[:, 21].values.astype("float32").reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = HospitalDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    test_dataset = HospitalDataset(
        torch.tensor(X_test), torch.tensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    model     = Model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # GPU
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/10 — loss: {epoch_loss/len(train_loader):.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # GPU
            output  = model(data)
            pred    = (output > 0).float()
            correct += (pred == target).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f"[TRAIN] Centralized model accuracy: {accuracy:.4f}")

    # ── Save model (CPU for portability) ─────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "global_model.pth")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"[TRAIN] Model saved → {model_path}")

    return accuracy


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    data_dir   = os.path.join(script_dir, "received_data")
    train_centralized_model(data_dir)