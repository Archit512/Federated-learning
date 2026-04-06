# Federated Learning UI - Architecture & Integration Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Federated Learning System                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       CENTRAL COORDINATOR                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Flower Server (Port 8080)          Server UI (Port 8501)           │
│  ┌──────────────────────┐          ┌──────────────────────┐         │
│  │ server_federated.py  │          │ server_ui.py         │         │
│  │                      │          │ (Streamlit)          │         │
│  │ • Aggregates models  │──────────│ • Monitor clients    │         │
│  │ • Coordinates rounds │          │ • View progress      │         │
│  │ • FedAvg/FedProx     │          │ • Analytics dashboard│         │
│  │ • TCP Port 8080      │          │                      │         │
│  └──────────────────────┘          └──────────────────────┘         │
│           ▲                                   ▲                      │
│           │                                   │                      │
└───────────┼───────────────────────────────────┼──────────────────────┘
            │                                   │
            │ (gRPC Port 8080)                 │ (Browser)
            │                                   │
     ┌──────┼───────────────────────────────────┼──────┐
     │      │                                   │      │
     ▼      ▼                                   ▼      ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   HOSPITAL A    │  │   HOSPITAL B    │  │   HOSPITAL C    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│                 │  │                 │  │                 │
│ Client Trainer  │  │ Client Trainer  │  │ Client Trainer  │
│ (Port 8081)     │  │ (Port 8082)     │  │ (Port 8083)     │
│                 │  │                 │  │                 │
│ Client UI       │  │ Client UI       │  │ Client UI       │
│ (Port 8502)     │  │ (Port 8503)     │  │ (Port 8504)     │
│                 │  │                 │  │                 │
│ Private Data    │  │ Private Data    │  │ Private Data    │
│ (Local CSV)     │  │ (Local CSV)     │  │ (Local CSV)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
     │                    │                      │
     └────────────────────┴──────────────────────┘
              (gRPC communication only)
                  NO DATA SHARED
```

---

## Component Breakdown

### 1. Flower Server (Backend Orchestration)
**File:** `ui/server_federated.py`

**Responsibilities:**
- Listen for client connections (port 8080)
- Coordinate federated learning rounds
- Aggregate model weights from clients
- Implement FedAvg or FedProx strategy
- Handle client evaluation

**Key Classes:**
```python
class FederatedLearningServer:
    - __init__()      # Initialize with strategy
    - start()         # Launch server
    - _create_strategy()  # FedAvg or FedProx
```

**Usage:**
```bash
python ui/server_federated.py --num_rounds 10 --strategy FedAvg
```

---

### 2. Server Monitoring UI (Real-time Dashboard)
**File:** `ui/server_ui.py`

**Responsibilities:**
- Display server status
- Show connected hospitals
- Monitor training progress
- Visualize global metrics
- Simulate training for testing

**Key Features:**
- 5 Main Tabs: Overview, Clients, Rounds, Aggregation, Analytics
- Real-time metrics visualization
- Round history tracking
- Client performance comparison

**Dependencies:**
- Streamlit (UI framework)
- Pandas (data handling)
- Custom utilities (shared_utils.py)

**Usage:**
```bash
streamlit run ui/server_ui.py
```

**Access:** http://localhost:8501

---

### 3. Hospital Client Trainer (Local Training)
**File:** `ui/client_federated.py`

**Responsibilities:**
- Connect to Flower server
- Receive global model weights
- Train locally on hospital data
- Return updated weights
- Evaluate on local test data

**Key Classes:**
```python
class HospitalClient(flwr.client.NumPyClient):
    - get_parameters()    # Extract model weights
    - set_parameters()    # Load received weights
    - fit()              # Local training
    - evaluate()         # Local evaluation
    - train()            # Training loop
    - test()             # Test loop
    - predict()          # Make predictions
```

**Usage:**
```bash
python ui/client_federated.py Hospital_A [csv_path] [server_address]
```

---

### 4. Hospital Client UI (Interactive Dashboard)
**File:** `ui/client_ui.py`

**Responsibilities:**
- Data exploration and statistics
- Local model training interface
- Prediction interface
- Performance monitoring

**Key Features:**
- 4 Main Tabs: Data Overview, Local Training, Predictions, Performance
- Visual data statistics
- Training progress monitoring
- Risk classification for predictions

**Dependencies:**
- Streamlit (UI framework)
- PyTorch (model training)
- Custom utilities (shared_utils.py)

**Usage:**
```bash
streamlit run ui/client_ui.py
```

**Access:** http://localhost:8502 (customizable)

---

### 5. Shared Utilities (Common Foundation)
**File:** `ui/shared_utils.py`

**Responsibilities:**
- Neural network model definition
- Data loading and splitting
- Model persistence (save/load)
- Training and evaluation logic
- Prediction interface
- Metrics handling

**Key Functions:**
```python
class Model(nn.Module)           # Neural network
class HospitalDataset(Dataset)   # PyTorch dataset
load_data()                      # Load CSV
create_data_loaders()           # Create dataloaders
evaluate_model()                # Evaluate
save_model()                    # Persistence
predict()                       # Inference
```

---

### 6. Configuration Management
**File:** `ui/config.py`

**Manages:**
- Server/client addresses and ports
- Training hyperparameters
- Model architecture details
- UI settings
- Data paths

**Key Variables:**
```python
FLOWERS_SERVER_CLIENT_ADDRESS = "127.0.0.1:8080"
NUM_ROUNDS = 10
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]
```

---

## Data Flow Diagram

### Training Round Flow

```
ROUND START
    │
    ├─→ Server broadcasts global model weights
    │
    ├─→ HOSPITAL A
    │   └─→ Receives weights
    │   └─→ Loads local training data
    │   └─→ Trains for 5 epochs
    │   └─→ Evaluates on test data
    │   └─→ Sends weights back to server
    │
    ├─→ HOSPITAL B (parallel with A)
    │   └─→ [same process]
    │
    ├─→ HOSPITAL C (parallel with A, B)
    │   └─→ [same process]
    │
    ├─→ HOSPITAL D (parallel with A, B, C)
    │   └─→ [same process]
    │
    ├─→ Server waits for all 4 clients
    │
    ├─→ Server aggregates weights (FedAvg)
    │   └─→ weighted_avg = Σ(n_k/n) * w_k
    │
    ├─→ Global model updated
    │
    ├─→ Server evaluation (optional)
    │
    └─→ ROUND COMPLETE → Next Round or Stop
```

### Data Privacy Guarantee

```
Hospital A (Local)          Hospital B (Local)
├─ raw patient data         ├─ raw patient data
│  (stays on premises)      │  (stays on premises)
│                           │
├─ model training          ├─ model training
│  (local computation)     │  (local computation)
│                           │
└─ model weights           └─ model weights
   (sent to server)           (sent to server)
   ↓                          ↓
   ┌────────────────────────────┐
   │                            │
   │ Central Server             │
   │                            │
   │ • Receives weights only    │
   │ • NO ACCESS to raw data    │
   │ • Aggregates weights       │
   │ • Broadcasts new model     │
   │                            │
   └────────────────────────────┘
```

---

## Integration Points

### 1. Client UI ↔ Client Trainer
**Connection:** Local Python process communication

```python
# client_federated.py creates HospitalClient
client = HospitalClient(model, train_loader, test_loader)

# Client UI can:
# - Trigger training in client_federated.py
# - Load trained models
# - Save/load metrics
```

### 2. Client ↔ Server (Flower gRPC)
**Protocol:** gRPC with NumPy serialization

```python
# In client_federated.py:
flwr.client.start_client(
    server_address="127.0.0.1:8080",  # Connect here
    client=client
)

# Flow:
# 1. Client sends get_parameters() → Server
# 2. Server sends fit request
# 3. Client trains locally
# 4. Client sends updated parameters → Server
```

### 3. Server ↔ Server UI
**Connection:** Streamlit session state + JSON files

```python
# server_ui.py (Streamlit)
st.session_state.round_history = [...]
st.session_state.aggregated_model = model

# Persistence:
save_round_summary(round_num, metrics_dict)
load_round_summary(round_num) → metrics_dict
```

### 4. Client UI ↔ Shared Utilities
**Connection:** Direct Python imports

```python
# In client_ui.py:
from shared_utils import:
    - Model                 # Neural network
    - load_data()           # Data loading
    - create_data_loaders() # DataLoader creation
    - evaluate_model()      # Evaluation
    - predict()             # Inference
```

---

## Execution Sequence (Complete Workflow)

### Setup Phase
1. User installs dependencies: `pip install -r ui/requirements.txt`
2. User prepares data: `python Data/split_balanced.py`
3. User configures settings in `config.py` (optional)

### Startup Phase
1. **Terminal 1:** Start server: `python ui/server_federated.py`
   - Server listens on port 8080
   - Waiting for 4 clients

2. **Terminal 2:** Start server UI: `streamlit run ui/server_ui.py`
   - Dashboard opens at localhost:8501
   - Shows "Waiting for clients"

3. **Terminal 3-4:** Hospital A training:
   - `streamlit run ui/client_ui.py` (UI at 8502)
   - `python ui/client_federated.py Hospital_A` (connects to server)

4. **Terminal 5-6:** Hospital B training:
   - `streamlit run ui/client_ui.py --server.port 8503`
   - `python ui/client_federated.py Hospital_B`

5. **Similar for Hospital C and D**

### Training Phase
1. Server detects all 4 clients connected
2. Server initiates Round 1
3. Each client:
   - Receives global model weights
   - Trains locally for 5 epochs
   - Sends back updated weights
4. Server aggregates all weights (FedAvg)
5. New global model created
6. Repeat for 10 rounds

### Monitoring Phase
- Server UI updates in real-time
- Shows client status, accuracy, loss
- Displays global model performance
- Generates performance visualizations

---

## File Dependencies Graph

```
config.py
├─ Used by: client_ui.py, server_ui.py, client_federated.py
└─ Provides: Constants, paths, hyperparameters

shared_utils.py
├─ Used by: client_ui.py, server_ui.py, client_federated.py
├─ Provides: Model, utility functions
└─ Requires: torch, pandas, sklearn

client_ui.py
├─ Uses: shared_utils.py, config.py
├─ Runs: Streamlit UI
└─ Requires: streamlit, torch, pandas

server_ui.py
├─ Uses: shared_utils.py, config.py
├─ Runs: Streamlit UI
└─ Requires: streamlit, torch, pandas

client_federated.py
├─ Uses: shared_utils.py, config.py
├─ Runs: Flower client
├─ Connects to: server_federated.py
└─ Requires: flwr, torch, pandas

server_federated.py
├─ Uses: shared_utils.py, config.py
├─ Runs: Flower server
├─ Receives connections from: client_federated.py
└─ Requires: flwr, torch
```

---

## Communication Protocols

### Flower Server ↔ Client (gRPC)
- **Port:** 8080 (default)
- **Protocol:** gRPC with Protocol Buffers
- **Data Format:** NumPy arrays (serialized)
- **Messages:**
  - `GetParameters` - Request model weights
  - `Fit` - Local training request
  - `Evaluate` - Evaluation request
  - `Parameters` - Model weights response

### Streamlit ↔ Browser (HTTP)
- **Port:** 8501 (Server UI), 8502+ (Client UIs)
- **Protocol:** WebSocket + HTTP
- **Data Format:** JSON + HTML
- **Updates:** Real-time via st.rerun() or callbacks

### File-Based (Persistence)
- **Models:** `ui/models/*.pth` (PyTorch format)
- **Metrics:** `ui/metrics/*.json` (JSON format)
- **Session State:** In-memory (Streamlit)

---

## Extension Points

### 1. Add New Aggregation Strategy
**File:** `ui/server_federated.py`
```python
def _create_strategy(self):
    if self.strategy == "CustomStrategy":
        return flwr.server.strategy.CustomStrategy(...)
```

### 2. Add New Model Architecture
**File:** `ui/shared_utils.py`
```python
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # New architecture
```

### 3. Add New UI Tab
**File:** `ui/client_ui.py`
```python
tab5 = st.tabs([..., "New Feature"])
with tab5:
    # New UI code
```

### 4. Add Differential Privacy
**File:** `ui/client_federated.py`
```python
def train(self, epochs=5):
    # Add DP noise to gradients
    # Use opacus library
```

---

## Scaling Considerations

### Current Limitations
- **Clients:** 4 hospitals (configurable)
- **Data Size:** Limited by local RAM
- **Model Size:** ~100KB (very small)
- **Network:** Assumes local network

### Scaling Improvements
1. **More Clients:**
   - Update `MIN_FIT_CLIENTS` in config.py
   - Create more client instances

2. **Larger Models:**
   - Modify Model class in shared_utils.py
   - Handle compression for faster transfer

3. **Remote Deployment:**
   - Use VPN or secure tunneling
   - Implement SSL/TLS
   - Add authentication

4. **Multiple Rounds:**
   - Modify `NUM_ROUNDS` in config.py
   - Track metrics across rounds

---

## Performance Metrics

### Expected Performance (Local Machine)
- **Model Size:** ~1.5 MB
- **Training Time (per round):** 2-5 seconds
- **Total 10 Rounds:** ~30-50 seconds
- **Accuracy (Balanced):** 75-90%
- **Memory Usage:** ~500 MB - 2 GB

### Optimization Tips
1. Use batch processing (BATCH_SIZE=16)
2. Use GPU if available
3. Reduce model complexity
4. Increase dropout for regularization
5. Use learning rate scheduling

---

## Testing & Validation

### Unit Tests
```python
# Test Model
model = Model()
x = torch.randn(16, 21)
y = model(x)
assert y.shape == (16, 1)

# Test Data Loading
X_train, X_test, y_train, y_test = load_data("path/to/csv")
assert X_train.shape[1] == 21
```

### Integration Tests
```python
# Test Client-Server Communication
# Run server and client in separate processes
# Verify weight exchange

# Test UI Rendering
# Run Streamlit app
# Check all tabs load correctly
```

### System Tests
```python
# Full end-to-end training
# 4 clients + server
# 10 rounds
# Verify convergence
```

---

## Troubleshooting Guide

### Connection Issues
```
Error: Connection refused on port 8080
→ Check if server is running
→ Check firewall rules
→ Try different port in config.py
```

### CUDA/Memory Issues
```
Error: CUDA out of memory
→ Use CPU: device = torch.device("cpu")
→ Reduce batch size
→ Reduce model size
```

### Port Conflicts
```
Error: Address already in use: port 8501
→ streamlit run ui/client_ui.py --server.port 8503
→ Check: netstat -ano | findstr :8501 (Windows)
```

### Data Issues
```
Error: FileNotFoundError for CSV
→ Run: python Data/split_balanced.py
→ Check data paths in config.py
```

---

## Summary

The Federated Learning UI system is composed of:
1. **Backend Components** (servers, trainers)
2. **Frontend Components** (Streamlit dashboards)
3. **Shared Libraries** (utilities, models)
4. **Configuration** (centralized settings)

All components work together to provide:
- ✅ Privacy-preserving distributed learning
- ✅ Real-time monitoring and control
- ✅ Flexible configuration
- ✅ Production-ready error handling
- ✅ Easy deployment and scaling

---

**Architecture Version:** 1.0  
**Last Updated:** April 2026
