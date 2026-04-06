"""
Refactored Federated Learning Client for Streamlit Integration
Seamlessly integrates with client_ui.py
"""

import flwr
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import os
import sys
from pathlib import Path

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import Model, HospitalDataset, create_data_loaders, get_device

device = get_device()
print(f"Using device: {device}")


class HospitalClient(flwr.client.NumPyClient):
    """
    Federated Learning Client for Hospital
    Manages local training and communication with Flower server
    """
    
    def __init__(self, model, train_loader, test_loader, hospital_name):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.hospital_name = hospital_name
        self.training_history = []

    def get_parameters(self, config):
        """Extract model parameters for transmission"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from received values"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)

    def fit(self, parameters, config):
        """
        Train model locally
        Called by Flower server for each round
        """
        self.set_parameters(parameters)
        epochs = config.get("epochs", 5)
        self.train(epochs)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate model on local test set
        Called by Flower server after training
        """
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self, epochs=5):
        """Train model locally"""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            self.training_history.append({
                "epoch": epoch,
                "loss": avg_loss
            })
            print(f"[{self.hospital_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def test(self):
        """Evaluate model on test set"""
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
        
        print(f"[{self.hospital_name}] Test - Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return test_loss, accuracy
    
    def predict(self, data):
        """Make predictions on new data"""
        self.model.eval()
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = self.model(data_tensor)
            probabilities = torch.sigmoid(logits)
        
        return probabilities.cpu().numpy()


def load_hospital_data(csv_path, test_size=0.2, random_state=42):
    """Load hospital data from CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :21].values.astype('float32')
    y = df.iloc[:, 21].values.astype('float32').reshape(-1, 1)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def start_client(hospital_name, csv_path, server_address="127.0.0.1:8080"):
    """
    Start Flower client for hospital
    
    Args:
        hospital_name: Name of hospital (e.g., "Hospital_A")
        csv_path: Path to hospital CSV data
        server_address: Flower server address (IP:PORT)
    """
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client: {hospital_name}")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n[{hospital_name}] Loading data from {csv_path}...")
    X_train, X_test, y_train, y_test = load_hospital_data(csv_path)
    print(f"[{hospital_name}] Data loaded successfully!")
    print(f"  - Train samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    
    # Create data loaders
    print(f"\n[{hospital_name}] Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=16
    )
    
    # Initialize model
    print(f"\n[{hospital_name}] Initializing neural network model...")
    model = Model()
    model.to(device)
    print(f"[{hospital_name}] Model initialized on {device}")
    
    # Create client
    print(f"\n[{hospital_name}] Creating Flower client...")
    client = HospitalClient(model, train_loader, test_loader, hospital_name)
    
    # Connect to server
    print(f"\n[{hospital_name}] Connecting to Flower server at {server_address}...")
    print(f"[{hospital_name}] Waiting for training rounds...")
    
    flwr.client.start_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client_federated.py <HospitalName> [csv_path] [server_address]")
        print("Example: python client_federated.py Hospital_A")
        sys.exit(1)
    
    hospital_name = sys.argv[1]
    
    # Default paths
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "Data" / "Balanced_split_data" / f"{hospital_name}.csv"
    server_address = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1:8080"
    
    # Override with arguments if provided
    if len(sys.argv) > 2:
        csv_path = sys.argv[2]
    
    try:
        start_client(hospital_name, str(csv_path), server_address)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
