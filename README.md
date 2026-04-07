# Privacy-Preserving Healthcare Diagnosis using Federated Learning

📖 **Additional Documentation:**  
➡️ [Federated Learning Technical Explanation](Federated_Learning.md)

---

## Overview

This project implements a decentralized machine learning architecture to predict heart disease risk using the CDC Heart Disease Health Indicators Dataset.
By leveraging Federated Learning (FL), the system trains a global PyTorch neural network across multiple isolated hospital clients without centralizing or sharing raw patient data.

The implementation evaluates Federated Averaging (FedAvg) under different real-world data distributions, including both IID-style (balanced) and Non-IID (unbalanced/skewed) partitions across four simulated hospital nodes.

## System Architecture

The repository contains a Flower-based federated learning setup coordinated by one central server and four edge clients.

### Server

- Runs Flower server on port `8089`.
- Executes 10 federated communication rounds.
- Uses a custom `FedAvg` strategy (`SisaStrategy`) to aggregate model weights.
- Archives global checkpoints after every round in `Server/global_history/`.
- Waits for all 4 clients before each round.

### Clients (Hospitals A, B, C, D)

- Load hospital slice CSVs from `Data/Balanced_split_data/Hospital_X/`.
- Train sequentially across 5 SISA slices per round (5 local epochs per slice).
- Save per-slice local checkpoints in `Client/Hospital_X/local_checkpoints/`.
- Return updated model parameters to the server.
- Evaluate locally and return loss/accuracy metrics.

### Neural Network Model

- Input layer: 21 features
- Hidden layer: 16 units + ReLU
- Output layer: 1 logit (binary classification)
- Loss function: `BCEWithLogitsLoss`

## Repository Structure

```text
Federated-learning/
├── Client/
│   ├── requirements.txt
│   ├── unlearning_utils.py
│   ├── Hospital_A/
│   │   ├── client_A.py
│   │   └── unlearn.py
│   ├── Hospital_B/
│   │   ├── client_B.py
│   │   └── unlearn.py
│   ├── Hospital_C/
│   │   ├── client_C.py
│   │   └── unlearn.py
│   └── Hospital_D/
│       ├── client_D.py
│       └── unlearn.py
├── Server/
│   ├── requirements.txt
│   ├── server_FedAvg.py
│   ├── server_FedProx.py
│   └── global_history/
├── automate_unlearning.py
└── Data/
    ├── split_balanced.py
    ├── split_unbalanced.py
    ├── Initial data/
    │   ├── heart_disease_data.xlsx
    │   ├── heart_disease_data.csv
    │   ├── xlxs_to_csv_file.py
    │   └── requirements.txt
    ├── Balanced_split_data/
    │   └── (balanced hospital CSVs)
    └── Unbalanced_split_data/
        └── (non-IID hospital CSVs)
```

## Data Pipeline

### 1. Source Data Transformation

Run `Data/Initial data/xlxs_to_csv_file.py` to regenerate the baseline CSV.

- Reads the raw XLSX file.
- Moves the target label column to the end of the dataframe.
- Writes the cleaned CSV without headers.

### 2. Dataset Splits for Hospitals

Generate edge datasets using one of these split styles:

- Balanced split (IID): Run `Data/split_balanced.py` to create randomized, equal-size partitions.
- Unbalanced split (Non-IID): Run `Data/split_unbalanced.py` to create skewed, demographic-isolated partitions.

Both scripts also create SISA slice folders:

- `Data/Balanced_split_data/Hospital_X/Hospital_X_slice_1..5.csv`
- `Data/Unbalanced_split_data/Hospital_X/Hospital_X_slice_1..5.csv`
- `Hospital_X_slice_map.json` metadata for unlearning.

### 3. Distribution

No manual CSV copy into `Client/Hospital_X/` is required. Clients read directly from the `Data/` split folders.

## Environment Setup

This project is designed to be deployed across multiple machines (for example, 1 server laptop and 4 client laptops).

### Server Setup

```powershell
cd Server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Client Setup (Run on each hospital machine)

```powershell
cd Client
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: If using Mac/Linux, activate with `source .venv/bin/activate`.

## Execution Order

### Step 1. Start Server

On the central server machine:

```powershell
python Server/server_FedAvg.py
```

The server listens on `0.0.0.0:8089`.

### Step 2. Configure Client IPs

In each client script (`client_A.py`, `client_B.py`, `client_C.py`, `client_D.py`), set:

```python
IP = "<SERVER_IP_ADDRESS>"
```

### Step 3. Start All 4 Clients

On each hospital machine, run its matching client script. Example for Hospital A:

```powershell
python Client/Hospital_A/client_A.py
```

Training begins automatically once the 4th client connects.

## Unlearning Workflow

1. Trigger unlearning on the target hospital (example):

```powershell
python Client/Hospital_A/unlearn.py --patient-id <PATIENT_ID> --data-style Balanced
```

2. Or use the coordinator script to remove + emit rollback signal:

```powershell
python automate_unlearning.py
```

3. Restart server and clients to retrain from rollback state.

Rollback signals are read from `Server/rollback_signal.json`, and global checkpoints are loaded from `Server/global_history/`.

## Evaluation

- Each client evaluates on its own local test split.
- Client metrics (loss and accuracy) are returned each round.
- For comparison, aggregate or average metrics across hospitals.

## Requirements

### Server

- `flwr==1.26.1`
- `torch>=2.0.0`

### Clients

- `flwr==1.26.1`
- `torch>=2.0.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.7.0`
- `requests>=2.31.0`

### Data Conversion

- `pandas>=1.0.0`
- `openpyxl>=3.0.0`

## Troubleshooting

- Clients cannot connect to server: Verify server IP is correct in each client script. Confirm port `8089` is open in the server firewall and all machines are on the same network.
- Training does not start: The server requires 4 available clients. Ensure all hospital clients are running before expecting federated rounds to proceed.
- Data file not found errors: Confirm `Data/Balanced_split_data/Hospital_X/` contains slice CSV files and `Hospital_X_slice_map.json`.

## Future Scope

The server currently aggregates weights with standard FedAvg.

- On balanced datasets, the global model converges smoothly.
- On unbalanced datasets, accuracy may fluctuate because of client drift.

Future iterations can explore FedProx to penalize local model drift and improve training stability under extreme Non-IID conditions.

## Citations and Acknowledgments

This project relies on the following open-source frameworks, datasets, and documentation:

- Dataset: Teboul, A. (2020). *Heart Disease Health Indicators Dataset* [Data set]. Kaggle. Compiled from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015.
- Federated Learning Framework: Flower Labs. *Tutorial Series: What is Federated Learning?* Flower Documentation. Available at: https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
- Algorithm (FedAvg): McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. In Artificial Intelligence and Statistics (AISTATS).
  
## License

This project is licensed under the MIT License. See `LICENSE` for details.
