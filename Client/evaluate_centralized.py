import sys
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# ── Same model architecture as federated clients ─────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

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


# ── Core evaluation function (importable) ────────────────────────────────────
def evaluate_hospital(model_path: str, csv_path: str) -> float:
    """Load global_model.pth and evaluate on the CSV. Returns accuracy (0–1)."""

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    X  = torch.tensor(df.iloc[:, :21].values.astype("float32"))
    y  = torch.tensor(df.iloc[:, 21].values.astype("float32").reshape(-1, 1))

    # Use only the test split (same 80/20 split as federated clients)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_dataset = HospitalDataset(X_test, y_test)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output  = model(data)
            pred    = (output > 0).float()
            correct += (pred == target).sum().item()

    return correct / len(test_loader.dataset)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_centralized.py <HospitalName>")
        print("Example: python evaluate_centralized.py Hospital_A")
        sys.exit(1)

    hospital_name = sys.argv[1]

    base_dir   = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "Centralized", "global_model.pth")
    csv_path   = os.path.join(base_dir, "..", "Data", "Balanced_split_data", f"{hospital_name}.csv")

    acc = evaluate_hospital(model_path, csv_path)
    print(f"{hospital_name} — Centralized Model Accuracy: {acc*100:.2f}%")