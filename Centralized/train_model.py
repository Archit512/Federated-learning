import glob
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRAIN] Using device: {device}")

MODEL_SAVE_PATH = "global_model.pth"


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


def train_centralized_model(data_dir: str) -> float:
    csv_files = glob.glob(f"{data_dir}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    filenames = [f.split("\\")[-1].split("/")[-1] for f in csv_files]
    print(f"[TRAIN] Merging {len(csv_files)} dataset(s): {filenames}")

    frames = [pd.read_csv(f, header=None) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    X = df.iloc[:, :21].values.astype("float32")
    y = df.iloc[:, 21].values.astype("float32").reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        HospitalDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=16, shuffle=True
    )
    test_loader = DataLoader(
        HospitalDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=16, shuffle=False
    )

    model = Model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n" + "=" * 52)
    print("  Centralized Training — 10 Epochs")
    print("=" * 52)

    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        bar = "█" * int((1 - avg_loss) * 20) if avg_loss < 1 else ""
        print(f"  Epoch {epoch+1:>2}/10 — loss: {avg_loss:.4f}  {bar}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (output > 0).float()
            correct += (pred == target).sum().item()

    accuracy = correct / len(test_loader.dataset)

    print("=" * 52)
    print(f"  Centralized Accuracy : {accuracy*100:.2f}%")
    print("=" * 52)
    print(f"\n  Copy this value into server_FedAvg.py / server_FedProx.py:")
    print(f"  CENTRALIZED_ACCURACY = {accuracy:.4f}\n")

    torch.save(model.cpu().state_dict(), MODEL_SAVE_PATH)
    print(f"[TRAIN] Model saved → {MODEL_SAVE_PATH}")

    return accuracy


if __name__ == "__main__":
    train_centralized_model("received_data")