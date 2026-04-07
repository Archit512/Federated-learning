import flwr
import torch
import torch.nn as nn
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import OrderedDict

CLIENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLIENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Client"))

from evaluate_centralized import evaluate_hospital
from plot_results import save_all_plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HOSPITAL_NAME = "Hospital_B"
FULL_DATA_PATH = PROJECT_ROOT / "Data" / "Balanced_split_data" / f"{HOSPITAL_NAME}.csv"

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

   
def load_data(file):
    df = pd.read_csv(file, header=None)
    X = df.iloc[:, :21].values.astype('float32')
    y = df.iloc[:, 21].values.astype('float32').reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class HospitalClient(flwr.client.NumPyClient):
    def __init__(self, model, test_loader, hospital_name, slice_dir, centralized_accuracy, results_dir):
        self.model = model
        self.test_loader = test_loader
        self.hospital_name = hospital_name
        self.slice_dir = slice_dir
        self.centralized_accuracy = centralized_accuracy
        self.results_dir = results_dir
        self.checkpoint_dir = CLIENT_DIR / "local_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.num_slices = 5
        self.slice_loaders, self.total_samples = self.load_slice_loaders()
        self.rounds = []
        self.local_accuracies = []

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})  # FIXED
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)  # FIXED

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("server_round", len(self.rounds) + 1)

        for slice_idx, slice_loader in enumerate(self.slice_loaders, start=1):
            self.train_on_slice(slice_loader)

            checkpoint_path = self.checkpoint_dir / f"round_{current_round}_slice_{slice_idx}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)

        return self.get_parameters(config={}), self.total_samples, {}

    def load_slice_loaders(self):
        loaders = []
        total_samples = 0

        for s in range(1, self.num_slices + 1):
            slice_file = self.slice_dir / f"{self.hospital_name}_slice_{s}.csv"
            if not slice_file.exists():
                raise FileNotFoundError(f"Missing slice file: {slice_file}")

            df = pd.read_csv(slice_file, header=None)
            X = torch.tensor(df.iloc[:, :21].values.astype("float32"))
            y = torch.tensor(df.iloc[:, 21].values.astype("float32").reshape(-1, 1))

            dataset = HospitalDataset(X, y)
            total_samples += len(dataset)
            loaders.append(DataLoader(dataset, batch_size=16, shuffle=True))

        return loaders, total_samples

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()

        round_number = len(self.rounds) + 1
        self.rounds.append(round_number)
        self.local_accuracies.append(float(accuracy))

        save_all_plots(
            hospital_name=self.hospital_name,
            rounds=self.rounds,
            local_accuracies=self.local_accuracies,
            centralized_accuracy=self.centralized_accuracy,
            out_dir=self.results_dir,
        )

        print(
            f"[{self.hospital_name}] Round {round_number} | "
            f"Local={accuracy*100:.2f}% | "
            f"Centralized={self.centralized_accuracy*100:.2f}%"
            if self.centralized_accuracy is not None
            else f"[{self.hospital_name}] Round {round_number} | Local={accuracy*100:.2f}%"
        )

        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train_on_slice(self, loader):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for _ in range(5):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def test(self):
        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = (output > 0).float()
                correct += (pred == target).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return test_loss, accuracy
    
    def predict(self, data):
        logit = self.model(data)
        prob = torch.sigmoid(logit)
        confidence = prob.item() * 100
        ans = 1 if confidence > 50 else 0
        print(f"Predicted Class: {ans} with confidence {confidence:.2f}%")


if __name__ == "__main__":
    print("[Hospital_B] Loading dataset...")
    _, X_test, _, y_test = load_data(FULL_DATA_PATH)

    print("[Hospital_B] Preparing test loader...")
    test_dataset  = HospitalDataset(X_test,  y_test)

    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    IP = "127.0.0.1" #IPv4 address of server machine
    results_dir = CLIENT_DIR / "results"
    slice_dir = PROJECT_ROOT / "Data" / "Balanced_split_data" / HOSPITAL_NAME

    model_path = PROJECT_ROOT / "Centralized" / "global_model.pth"
    csv_path = FULL_DATA_PATH
    centralized_accuracy = None
    try:
        centralized_accuracy = evaluate_hospital(str(model_path), str(csv_path))
        print(f"[{HOSPITAL_NAME}] Centralized reference accuracy: {centralized_accuracy*100:.2f}%")
    except Exception as exc:
        print(f"[{HOSPITAL_NAME}] WARNING: centralized accuracy unavailable ({exc})")

    print("[Hospital_B] Initializing model...")
    model = Model()
    print(f"[Hospital_B] Connecting to server at {IP}:8089...")
    
    client = HospitalClient(model, test_loader, HOSPITAL_NAME, slice_dir, centralized_accuracy, results_dir)
    print("[Hospital_B] Client started. Waiting for federated rounds...")
    flwr.client.start_client(server_address=f"{IP}:8089", client=client.to_client())
