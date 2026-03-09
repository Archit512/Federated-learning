import flwr
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import OrderedDict

DataFile = "Hospital_C.csv"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
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

   
def load_data(file):
    df = pd.read_csv(file, header=None)

    X = df.iloc[:, :-2].values.astype('float32')

    y = df.iloc[:, -2:].values.astype('float32')

    return train_test_split(X, y, test_size=0.2, random_state=42)

class HospitalClient(flwr.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self,config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def train(self):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(5): 
            for data, target in self.train_loader:
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
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = (output > 0).float()
                correct += (pred == target).all(dim=1).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return test_loss, accuracy
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(DataFile)

    train_dataset = HospitalDataset(X_train, y_train)
    test_dataset = HospitalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    IP = "" #Add here the server Laptop's actual IP

    model = Model()
    print(f"Connecting to Server at {IP}:8080...")
    
    client = HospitalClient(model, train_loader, test_loader)
    flwr.client.start_numpy_client(server_address=f"{IP}:8080", client=client.to_client())