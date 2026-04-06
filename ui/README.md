# Federated Learning Streamlit UI - Setup & Usage Guide

## Overview

This guide explains how to set up and use the Streamlit user interfaces for the Federated Learning healthcare system. The system includes:

- **Client UI**: For hospital staff to manage local training and predictions
- **Server UI**: For administrators to monitor federated learning coordination

## Architecture

```
Federated Learning System
├── Streamlit Server UI (Port 8501)
│   └── Monitors all clients and aggregation
├── Flower Server (Port 8080)
│   └── Coordinates federated learning
├── Streamlit Client UI #1 - Hospital A (Port 8502)
├── Streamlit Client UI #2 - Hospital B (Port 8503)
├── Streamlit Client UI #3 - Hospital C (Port 8504)
└── Streamlit Client UI #4 - Hospital D (Port 8505)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- At least 4GB RAM (8GB recommended)

### Step 1: Install UI Dependencies

```bash
cd Federated-learning
pip install -r ui/requirements.txt
```

This installs:
- Streamlit (UI framework)
- Flower (Federated Learning)
- PyTorch (Deep Learning)
- Pandas & NumPy (Data processing)

### Step 2: Verify Data Files

Ensure the hospital data files exist:
```
Data/Balanced_split_data/
├── Hospital_A.csv
├── Hospital_B.csv
├── Hospital_C.csv
└── Hospital_D.csv
```

For balanced data, run:
```bash
python Data/split_balanced.py
```

For unbalanced data, run:
```bash
python Data/split_unbalanced.py
```

## Usage

### Option 1: Full Federated Learning Setup (4 Clients + Server)

This simulates a realistic scenario with hospitals and a central coordinator.

#### Terminal 1: Start Flower Server
```bash
python ui/server_federated.py --num_rounds 10 --strategy FedAvg
```

Output:
```
============================================================
Starting Federated Learning Server
============================================================
Strategy: FedAvg
Number of Rounds: 10
Waiting for 4 clients to connect...
```

#### Terminal 2: Start Server Monitoring UI
```bash
streamlit run ui/server_ui.py
```

Opens at: http://localhost:8501

**Server UI Features:**
- Monitor server status and connected clients
- View real-time training progress
- See aggregation results after each round
- Analyze global model performance
- View metrics across all rounds

#### Terminals 3-6: Start Client UIs and Federated Clients

For Hospital A:
```bash
# Terminal 3: Client UI
streamlit run ui/client_ui.py --logger.level=error

# Terminal 4: Federated Client
python ui/client_federated.py Hospital_A
```

For Hospital B:
```bash
# Terminal 5: Client UI
streamlit run ui/client_ui.py --logger.level=error

# Terminal 6: Federated Client
python ui/client_federated.py Hospital_B
```

Similar setup for Hospital C and D.

**Client UI Features:**
- Load and explore hospital data
- Train local models
- Make predictions
- Monitor local performance
- View training history

### Option 2: Client-Only Setup (Testing Single Hospital)

If you want to test a single hospital's UI without full federated setup:

```bash
streamlit run ui/client_ui.py
```

Features available:
- Data exploration
- Local model training
- Predictions on local data
- Performance monitoring

### Option 3: Server-Only Monitoring (UI Demo)

To test the server UI with simulated training:

```bash
streamlit run ui/server_ui.py
```

Features available:
- Simulated client connections
- Manual simulation of training rounds
- Analytics dashboard
- Metrics visualization

## Client UI Walkthrough

### Tab 1: Data Overview
1. Select hospital from sidebar
2. Click "Load Hospital Data"
3. View data statistics and distributions

### Tab 2: Local Training
1. Click "Initialize New Model"
2. Configure training parameters:
   - Epochs (1-20)
   - Batch Size (8-64)
   - Learning Rate
3. Click "Start Local Training"
4. Monitor training progress and loss

### Tab 3: Predictions
1. Set prediction threshold
2. Click "Generate Predictions"
3. View sample predictions with confidence scores
4. See risk levels (Low/Moderate/High)

### Tab 4: Performance
- View latest training metrics
- Check accuracy and loss
- See training configuration

## Server UI Walkthrough

### Tab 1: Overview
- Server status and metrics
- Global model information
- Quick actions (initialize, save, load models)

### Tab 2: Clients
- View connected hospitals
- Check client status
- Manage client connections

### Tab 3: Round Progress
- Monitor training rounds
- View client metrics per round
- Simulate rounds
- See accuracy/loss trends

### Tab 4: Aggregation
- View aggregation strategy details
- Understand FedAvg vs FedProx
- Perform manual aggregation testing

### Tab 5: Analytics
- Detailed round analysis
- Client performance comparison
- Comparative visualizations

## Configuration

### Hospital Selection
In the client UI sidebar, select which hospital to work with:
- Hospital_A
- Hospital_B
- Hospital_C
- Hospital_D

### Server Address
Default: `127.0.0.1:8080`

To change, edit config.py:
```python
FLOWERS_SERVER_CLIENT_ADDRESS = "127.0.0.1:8080"
```

### Training Parameters
In config.py, modify:
```python
NUM_ROUNDS = 10              # Federated learning rounds
EPOCHS_PER_ROUND = 5         # Local training epochs
BATCH_SIZE = 16              # Training batch size
LEARNING_RATE = 0.001        # Adam optimizer learning rate
PREDICTION_THRESHOLD = 0.5   # Classification threshold
FEDPROX_MU = 0.05            # FedProx proximal term
```

## Data Flow

### Client Training Flow
```
1. Load local hospital data (CSV)
2. Split into train/test (80/20)
3. Initialize neural network model
4. Train locally for N epochs
5. Evaluate on local test set
6. Save model and metrics
7. Make predictions on new data
```

### Federated Training Flow
```
Round 1:
  1. Server broadcasts model to all clients
  2. Each client trains locally (5 epochs)
  3. Clients return updated weights to server
  4. Server aggregates weights (FedAvg)
  5. New global model created
  6. Server evaluation
  
Round 2-10: Repeat...
```

## Model Architecture

```
Input Layer: 21 features
  ↓
Hidden Layer: 16 units + ReLU
  ↓
Output Layer: 1 logit (binary classification)
```

Loss Function: BCEWithLogitsLoss
Optimizer: Adam (lr=0.001)

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit
```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
streamlit run ui/server_ui.py --server.port 8502
```

### Issue: "Connection refused" when clients try to connect to server
**Solution:**
1. Ensure server is running: `python ui/server_federated.py`
2. Check server address in client code
3. Verify firewall allows port 8080

### Issue: "CUDA out of memory"
**Solution:**
```python
# In shared_utils.py or client code, use CPU:
device = torch.device("cpu")
```

### Issue: Data files not found
**Solution:**
1. Verify data exists in `Data/Balanced_split_data/`
2. Run `python Data/split_balanced.py`
3. Check file paths in config.py

## Performance Optimization

### For Faster Training
- Reduce EPOCHS_PER_ROUND in config.py
- Increase BATCH_SIZE
- Use GPU if available

### For Better Model Quality
- Increase EPOCHS_PER_ROUND
- Decrease BATCH_SIZE
- Increase NUM_ROUNDS
- Lower LEARNING_RATE

## Model Persistence

### Save Model
In Client UI → Training tab → Click "Start Local Training"
Model automatically saves to: `ui/models/Hospital_A_model.pth`

### Load Model
In Client UI → Performance tab → View saved metrics

### Global Model
In Server UI → Overview tab → Click "Save Global Model"
Saves to: `ui/models/global_model.pth`

## Data Privacy

The system ensures privacy:
- ✅ Raw data never leaves the hospital
- ✅ Only model weights transmitted over network
- ✅ Server aggregates weights without seeing data
- ✅ Each hospital trains independently

## Security Recommendations

1. **Network Security**: Run on secure, closed networks
2. **Access Control**: Restrict UI access with authentication
3. **Data Encryption**: Use HTTPS for production
4. **Model Protection**: Regularly backup trained models
5. **Audit Logging**: Keep training logs for compliance

## Advanced Usage

### Using FedProx Strategy
```bash
python ui/server_federated.py --strategy FedProx --num_rounds 10
```

FedProx is recommended for:
- Non-IID (unbalanced) data distributions
- High variation in local models
- Reducing client drift

### Custom Training Scenarios
Edit `ui/client_federated.py` to:
- Add custom preprocessing
- Implement different model architectures
- Add regularization techniques

### Metrics Analysis
Access saved metrics in `ui/metrics/`:
- `Hospital_A_metrics.json` - Local metrics
- `round_01_metrics.json` - Round-specific metrics
- `server_metrics.json` - Global aggregation metrics

## File Structure

```
ui/
├── shared_utils.py          # Common utilities
├── config.py                # Configuration constants
├── client_ui.py             # Hospital client interface
├── server_ui.py             # Server monitoring interface
├── client_federated.py      # Federated client implementation
├── server_federated.py      # Federated server implementation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Saved models directory
│   ├── Hospital_A_model.pth
│   ├── Hospital_B_model.pth
│   ├── Hospital_C_model.pth
│   ├── Hospital_D_model.pth
│   └── global_model.pth
└── metrics/                # Training metrics
    ├── Hospital_A_metrics.json
    ├── round_01_metrics.json
    └── server_metrics.json
```

## Support & Documentation

For more information:
- Federated Learning Concepts: See `Federated_Learning.md`
- Project Overview: See `README.md`
- Requirements: See `REQUIREMENTS.md`

## Performance Metrics

### Expected Results (Balanced Data)
- Individual Client Accuracy: 70-85%
- Global Model Accuracy: 75-90%
- Communication Rounds: 10
- Local Training Time: ~2-5 seconds per round
- Total Training Time: ~2-5 minutes

### Expected Results (Unbalanced Data)
- Individual Client Accuracy: 60-75%
- Global Model Accuracy: 65-80%
- Better with FedProx strategy
- May require more rounds for convergence

## Future Enhancements

Potential improvements:
- [ ] Differential Privacy integration
- [ ] Secure aggregation
- [ ] Multi-model ensembles
- [ ] Cross-hospital federated learning
- [ ] Real-time monitoring dashboards
- [ ] Automated hyperparameter tuning
- [ ] Model versioning and rollback

## License

See LICENSE file in the project root.

---

**Last Updated:** April 2026
**Version:** 1.0
**Status:** Production Ready
