"""
Shared utilities for Federated Learning UI (Client and Server)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pickle


class Model(nn.Module):
    """Neural Network Model - Same architecture as federated learning system"""
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
    """Dataset class for hospital data"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(csv_path, test_size=0.2, random_state=42):
    """
    Load and split hospital data
    Returns: X_train, X_test, y_train, y_test
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :21].values.astype('float32')
    y = df.iloc[:, 21].values.astype('float32').reshape(-1, 1)
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    """
    Create PyTorch DataLoaders for training and testing
    """
    train_dataset = HospitalDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )
    test_dataset = HospitalDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_device():
    """Get available device (CUDA or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, save_path):
    """Save model state dict"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_model(model_path, device='cpu'):
    """Load model from state dict"""
    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def predict(model, features, threshold=0.5, device='cpu'):
    """
    Make predictions on input features
    Returns: predictions, probabilities, confidence scores
    """
    model.eval()
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set
    Returns: accuracy, loss
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = (output > 0).float()
            correct += (pred == target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    return accuracy, test_loss


def get_data_statistics(X_train, X_test, y_train, y_test):
    """Get statistics about the data"""
    stats = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X_train) + len(X_test),
        "positive_class_train": int((y_train == 1).sum()),
        "negative_class_train": int((y_train == 0).sum()),
        "positive_class_test": int((y_test == 1).sum()),
        "negative_class_test": int((y_test == 0).sum()),
        "feature_count": X_train.shape[1],
        "train_positive_ratio": float((y_train == 1).sum() / len(y_train)),
        "test_positive_ratio": float((y_test == 1).sum() / len(y_test)),
    }
    return stats


def save_metrics(metrics, save_path):
    """Save metrics dictionary to pickle file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(metrics, f)


def load_metrics(save_path):
    """Load metrics from pickle file"""
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    return {}


def format_metrics(accuracy, loss):
    """Format metrics for display"""
    return {
        "accuracy": f"{accuracy*100:.2f}%",
        "loss": f"{loss:.4f}"
    }
