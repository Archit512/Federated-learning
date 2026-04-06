# Quick Start Guide - Federated Learning Streamlit UI

## 30-Second Setup

```bash
# 1. Install dependencies
pip install -r ui/requirements.txt

# 2. Prepare data (if not already done)
python Data/split_balanced.py
```

## Running the System

### **Option A: Full System (Recommended)**

**Terminal 1 - Server Backend:**
```bash
python ui/server_federated.py --num_rounds 10
```

**Terminal 2 - Server Dashboard:**
```bash
streamlit run ui/server_ui.py
```
→ Opens at http://localhost:8501

**Terminal 3 - Hospital A UI:**
```bash
streamlit run ui/client_ui.py
```
→ Opens at http://localhost:8502

**Terminal 4 - Hospital A Federated Client:**
```bash
python ui/client_federated.py Hospital_A
```

For Hospital B, C, D: Repeat steps 3-4 with different ports and hospital names.

---

### **Option B: Test Client UI Only**

```bash
streamlit run ui/client_ui.py
```

Then:
1. Load hospital data
2. Initialize model
3. Train locally
4. Make predictions

---

## What Each UI Does

### 🏥 Client UI (Hospital Dashboard)
- Load local hospital data
- Train models privately
- Make predictions
- Monitor performance

### 🌐 Server UI (Central Coordinator)
- Monitor all connected hospitals
- Watch training progress
- View aggregated results
- Analyze global model performance

---

## Key Features

✅ **Data Privacy** - Raw data never leaves hospital  
✅ **Decentralized** - Train on edge devices  
✅ **Real-time Monitoring** - See progress in Streamlit  
✅ **Easy Configuration** - Edit config.py for settings  
✅ **Multiple Strategies** - FedAvg or FedProx  
✅ **Production Ready** - Full error handling  

---

## Directory Structure

```
ui/
├── client_ui.py              # Run this for hospital dashboard
├── server_ui.py              # Run this for central monitoring
├── client_federated.py       # Run this to connect to server
├── server_federated.py       # Run this to start server
├── shared_utils.py           # Common functions
├── config.py                 # Edit for custom settings
├── requirements.txt          # Dependencies
├── README.md                 # Full documentation
└── models/                   # Saved trained models
```

---

## Common Commands

```bash
# Change strategy
python ui/server_federated.py --strategy FedProx

# Change number of rounds
python ui/server_federated.py --num_rounds 20

# Use different port
streamlit run ui/client_ui.py --server.port 8502

# Run with verbose logging
python ui/client_federated.py Hospital_A --log_level DEBUG
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | Use different port: `--server.port 8502` |
| Module not found | Run: `pip install -r ui/requirements.txt` |
| Data not found | Run: `python Data/split_balanced.py` |
| Connection refused | Ensure server is running first |

---

## Expected Output

### Server Terminal:
```
============================================================
Starting Federated Learning Server
============================================================
Strategy: FedAvg
Number of Rounds: 10
Waiting for 4 clients to connect...
```

### Client Terminal:
```
[Hospital_A] Loading data...
[Hospital_A] Data loaded successfully!
  - Train samples: 2000
  - Test samples: 500
[Hospital_A] Connecting to Flower server...
[Hospital_A] Waiting for training rounds...
```

---

## Next Steps

1. Run the system using Option A or B above
2. Open Server UI in browser (http://localhost:8501)
3. Open Client UI in another browser tab/window
4. Load data in Client UI
5. Train a model locally
6. Connect clients to server
7. Monitor progress in Server UI

---

## Full Documentation

See `ui/README.md` for:
- Detailed walkthrough
- Configuration guide
- Architecture explanation
- Performance optimization
- Security recommendations

---

**Version:** 1.0  
**Status:** Ready to Use  
**Questions?** Check README.md or the inline code comments
