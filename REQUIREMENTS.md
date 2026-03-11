# Requirements Documentation

This document describes the dependencies required for the Federated Learning project.

## Architecture Overview

This federated learning system is designed to run on **5 separate devices**:

- **1 Server Device**: Runs the federated learning coordinator (`Server/server.py`)
  - Aggregates model updates from all clients
  - Coordinates 10 rounds of federated learning
  - Requires all 4 clients to be available before training begins
  
- **4 Client Devices (Hospitals)**: Each hospital trains on its local data
  - `Hospital_A` - Client with local medical data
  - `Hospital_B` - Client with local medical data
  - `Hospital_C` - Client with local medical data
  - `Hospital_D` - Client with local medical data

Each device should install its respective requirements (Server or Client) based on its role in the federated network.

## Core Dependencies

### 1. Flower (flwr)
- **Version**: >= 1.0.0
- **Purpose**: Federated learning framework used to coordinate distributed machine learning across multiple clients (hospitals)
- **Used in**: `Server/server.py`, `Client/Hospital_A/client_A.py`, `Client/Hospital_B/client_B.py`, `Client/Hospital_C/client_C.py`, `Client/Hospital_D/client_D.py`
- **Key Features**:
  - Implements federated averaging (FedAvg) strategy
  - Manages client-server communication
  - Coordinates training rounds across distributed nodes

### 2. PyTorch (torch)
- **Version**: >= 2.0.0
- **Purpose**: Deep learning framework for building and training neural networks
- **Used in**: `Client/Hospital_A/client_A.py`, `Client/Hospital_B/client_B.py`, `Client/Hospital_C/client_C.py`, `Client/Hospital_D/client_D.py`
- **Key Features**:
  - Neural network model definition
  - Training and optimization
  - DataLoader utilities for batch processing

### 3. Pandas
- **Version**: >= 2.0.0
- **Purpose**: Data manipulation and analysis library
- **Used in**: `Client/Hospital_A/client_A.py`, `Client/Hospital_B/client_B.py`, `Client/Hospital_C/client_C.py`, `Client/Hospital_D/client_D.py`, `Data/csv_file.py`
- **Key Features**:
  - Loading CSV data files (e.g., HospitalA.csv)
  - Data preprocessing
  - DataFrame operations

### 4. Scikit-learn (sklearn)
- **Version**: >= 1.3.0
- **Purpose**: Machine learning utilities
- **Used in**: `Client/Hospital_A/client_A.py`, `Client/Hospital_B/client_B.py`, `Client/Hospital_C/client_C.py`, `Client/Hospital_D/client_D.py`
- **Key Features**:
  - Train-test data splitting (`train_test_split`)
  - Data preprocessing utilities
  - Model evaluation metrics

### 5. OpenPyXL
- **Version**: >= 3.1.0
- **Purpose**: Excel engine required by pandas to read `.xlsx` files
- **Used in**: `Data/csv_file.py`

## Installation

The project has separate requirements files for the server and client components. Install the appropriate requirements on each device based on its role.

### For Server Device (1 device):

```bash
cd Server
pip install -r requirements.txt
```

This installs:
- flwr

### For Client Devices (4 devices - Hospital_A, Hospital_B, Hospital_C, Hospital_D):

```bash
cd Client
pip install -r requirements.txt
```

This installs:
- flwr
- torch
- pandas
- scikit-learn

**Note**: Each of the 4 hospital client devices should install the client requirements.

### Or install individually:

```bash
# Server dependencies
pip install flwr

# Client dependencies (includes server dependencies)
pip install flwr torch pandas scikit-learn
```

## Virtual Environment (Recommended)

It's recommended to use a virtual environment on each device to manage dependencies:

### For Server Device:

```bash
cd Server
python -m venv .venv

# Activate on Windows
.venv\Scripts\Activate.ps1

# Install server requirements
pip install -r requirements.txt
```

### For Each Client Device (repeat on all 4 hospital devices):

```bash
cd Client
python -m venv .venv

# Activate on Windows
.venv\Scripts\Activate.ps1

# Install client requirements
pip install -r requirements.txt
```

## System Requirements

### Per Device

- **Python**: 3.8 or higher
- **Operating System**: Windows
- **RAM**: 
  - Server: 2GB minimum
  - Client: 4GB minimum (8GB+ recommended for training)
- **GPU**: Optional for clients (CUDA-compatible GPU for faster training with PyTorch)
- **Network**: All 5 devices must be able to communicate over the network
  - Server runs on port 8080 by default
  - Clients must be able to reach the server address

### Total Setup

- **5 devices total**: 1 server + 4 clients (hospitals)
- **Network connectivity**: Required between all devices
- **Synchronized startup**: All 4 clients must be available to server before training begins