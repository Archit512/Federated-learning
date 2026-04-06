"""
Streamlit UI for Federated Learning Server
Allows monitoring of:
- Connected clients
- Training progress across federated rounds
- Model aggregation results
- Global model performance
- Communication metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import (
    Model, get_device, save_model, load_model,
    create_data_loaders, evaluate_model, format_metrics
)
from config import (
    HOSPITALS, NUM_ROUNDS, MIN_FIT_CLIENTS,
    MIN_AVAILABLE_CLIENTS, FEDPROX_MU, PREDICTION_THRESHOLD
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Federated Learning Server",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .status-active { color: #09ab3b; }
    .status-inactive { color: #ff2b2b; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "server_running" not in st.session_state:
    st.session_state.server_running = False
if "connected_clients" not in st.session_state:
    st.session_state.connected_clients = []
if "current_round" not in st.session_state:
    st.session_state.current_round = 0
if "round_history" not in st.session_state:
    st.session_state.round_history = []
if "aggregated_model" not in st.session_state:
    st.session_state.aggregated_model = None
if "global_metrics" not in st.session_state:
    st.session_state.global_metrics = []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_server_model_path():
    """Get path to save aggregated model"""
    base_path = Path(__file__).parent.parent / "ui" / "models"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / "global_model.pth"

def get_server_metrics_path():
    """Get path to save server metrics"""
    base_path = Path(__file__).parent.parent / "ui" / "metrics"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / "server_metrics.json"

def get_round_metrics_path(round_num):
    """Get path to save round-specific metrics"""
    base_path = Path(__file__).parent.parent / "ui" / "metrics"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / f"round_{round_num:02d}_metrics.json"

def aggregate_models(models, weights=None):
    """
    Aggregate multiple models using FedAvg
    models: list of Model objects
    weights: optional list of weights for each model
    """
    if not models:
        return None
    
    # If no weights provided, use equal weights
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Initialize aggregated state dict
    aggregated_state = None
    
    for model, weight in zip(models, weights):
        model_state = model.state_dict()
        
        if aggregated_state is None:
            aggregated_state = {}
            for key in model_state.keys():
                aggregated_state[key] = weight * model_state[key].clone()
        else:
            for key in model_state.keys():
                aggregated_state[key] += weight * model_state[key]
    
    # Create new model and load aggregated state
    aggregated_model = Model()
    aggregated_model.load_state_dict(aggregated_state)
    
    return aggregated_model

def simulate_client_training(client_name):
    """Simulate client training and return updated model"""
    device = get_device()
    model = Model()
    model.to(device)
    
    # Simulate local training
    # In production, this would connect to actual clients
    return model

def save_round_summary(round_num, summary):
    """Save summary of a training round"""
    path = get_round_metrics_path(round_num)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

def load_round_summary(round_num):
    """Load summary of a training round"""
    path = get_round_metrics_path(round_num)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

# ============================================================================
# MAIN UI
# ============================================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext x='50' y='70' font-size='70' text-anchor='middle'%3E🌐%3C/text%3E%3C/svg%3E", width=60)
with col2:
    st.title("Federated Learning Server")
    st.caption("Central Coordinator for Privacy-Preserving Training")

# ============================================================================
# SIDEBAR - SERVER CONTROL
# ============================================================================
with st.sidebar:
    st.header("⚙️ Server Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟢 Start Server", key="start_server", use_container_width=True):
            st.session_state.server_running = True
            st.session_state.current_round = 0
            st.success("Server started")
    
    with col2:
        if st.button("🛑 Stop Server", key="stop_server", use_container_width=True):
            st.session_state.server_running = False
            st.warning("Server stopped")
    
    st.divider()
    
    # Server Status
    st.subheader("Status")
    if st.session_state.server_running:
        st.success("✓ Server Running")
    else:
        st.error("✗ Server Inactive")
    
    st.divider()
    
    # Configuration
    st.subheader("Configuration")
    strategy = st.radio("Aggregation Strategy", ["FedAvg", "FedProx"])
    
    num_rounds = st.number_input(
        "Number of Rounds",
        min_value=1,
        max_value=100,
        value=NUM_ROUNDS
    )
    
    if strategy == "FedProx":
        mu = st.number_input(
            "FedProx μ (mu)",
            min_value=0.0,
            max_value=1.0,
            value=FEDPROX_MU,
            step=0.01
        )
    
    st.divider()
    
    st.subheader("Connected Clients")
    for hospital in HOSPITALS:
        is_connected = hospital in st.session_state.connected_clients
        status = "✓ Connected" if is_connected else "✗ Not Connected"
        st.write(f"{hospital}: {status}")

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Overview",
    "👥 Clients",
    "📊 Round Progress",
    "🤝 Aggregation",
    "📈 Analytics"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Server Overview")
    
    # Server Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "Running" if st.session_state.server_running else "Stopped"
        st.metric("Server Status", status)
    
    with col2:
        st.metric("Current Round", st.session_state.current_round)
    
    with col3:
        st.metric("Total Rounds", num_rounds)
    
    with col4:
        st.metric("Connected Clients", len(st.session_state.connected_clients))
    
    st.divider()
    
    # Global Model Info
    st.subheader("🤖 Global Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.aggregated_model is not None:
            st.success("✓ Global Model Available")
            
            # Model parameters
            total_params = sum(p.numel() for p in st.session_state.aggregated_model.parameters())
            st.metric("Total Parameters", f"{total_params:,}")
        else:
            st.info("No aggregated model yet")
    
    with col2:
        model_path = get_server_model_path()
        if model_path.exists():
            file_size = os.path.getsize(model_path) / 1024  # KB
            st.metric("Model Size", f"{file_size:.2f} KB")
    
    st.divider()
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Initialize Global Model", key="init_global", use_container_width=True):
            device = get_device()
            model = Model()
            model.to(device)
            st.session_state.aggregated_model = model
            st.success("Global model initialized")
    
    with col2:
        if st.button("Save Global Model", key="save_global", use_container_width=True):
            if st.session_state.aggregated_model is not None:
                path = get_server_model_path()
                save_model(st.session_state.aggregated_model, str(path))
                st.success(f"Model saved to {path}")
            else:
                st.warning("No model to save")
    
    with col3:
        if st.button("Load Global Model", key="load_global", use_container_width=True):
            path = get_server_model_path()
            if path.exists():
                device = get_device()
                model = load_model(str(path), device)
                st.session_state.aggregated_model = model
                st.success("Model loaded")
            else:
                st.warning("No saved model found")

# ============================================================================
# TAB 2: CLIENTS
# ============================================================================
with tab2:
    st.header("Connected Clients")
    
    # Client Management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Manage Client Connections")
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_clients", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    # Client Table
    client_data = []
    for hospital in HOSPITALS:
        is_connected = hospital in st.session_state.connected_clients
        status = "Connected" if is_connected else "Disconnected"
        client_data.append({
            "Hospital": hospital,
            "Status": status,
            "Dataset": f"{hospital}.csv",
            "Action": "Disconnect" if is_connected else "Connect"
        })
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write("### Client Status")
        for client in client_data:
            color = "green" if client["Status"] == "Connected" else "red"
            st.write(
                f"<span style='color:{color}'>● {client['Hospital']}: {client['Status']}</span>",
                unsafe_allow_html=True
            )
    
    with col2:
        st.write("### Simulated Clients")
        refreshed = False
    
    with col3:
        st.write("### Actions")
    
    st.divider()
    
    # Client Details
    st.subheader("Client Details")
    selected_client = st.selectbox("Select Client", HOSPITALS)
    
    if selected_client:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Hospital:** {selected_client}")
            st.write(f"**Status:** {'Connected' if selected_client in st.session_state.connected_clients else 'Not Connected'}")
            st.write("**Data Path:** Data/Balanced_split_data/{selected_client}.csv")
        
        with col2:
            st.write("**Configuration:**")
            st.write("- Training Epochs: 5")
            st.write("- Batch Size: 16")
            st.write("- Learning Rate: 0.001")

# ============================================================================
# TAB 3: ROUND PROGRESS
# ============================================================================
with tab3:
    st.header("Training Round Progress")
    
    st.subheader("Simulate Federated Training Rounds")
    
    if st.session_state.server_running:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Simulate Next Round", key="simulate_round", use_container_width=True):
                st.info("Simulating federated round...")
                
                # Simulate client connections
                if st.session_state.current_round == 0:
                    st.session_state.connected_clients = HOSPITALS.copy()
                
                # Simulate training
                round_num = st.session_state.current_round + 1
                
                # Create progress bar
                progress_bar = st.progress(0)
                round_status = st.empty()
                
                # Simulate client training
                models = []
                client_metrics = {}
                
                for i, hospital in enumerate(HOSPITALS):
                    round_status.write(f"Training {hospital}...")
                    progress_bar.progress((i + 1) / len(HOSPITALS))
                    
                    # Simulate training results
                    model = Model()
                    models.append(model)
                    client_metrics[hospital] = {
                        "accuracy": np.random.uniform(0.7, 0.95),
                        "loss": np.random.uniform(0.2, 0.5),
                        "samples_trained": np.random.randint(1000, 3000)
                    }
                
                # Aggregate models
                round_status.write("Aggregating models...")
                progress_bar.progress(0.9)
                
                aggregated_model = aggregate_models(models)
                st.session_state.aggregated_model = aggregated_model
                st.session_state.current_round = round_num
                
                # Save round summary
                round_summary = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "clients_trained": len(models),
                    "client_metrics": client_metrics,
                    "global_accuracy": float(np.mean([m["accuracy"] for m in client_metrics.values()])),
                    "global_loss": float(np.mean([m["loss"] for m in client_metrics.values()])),
                    "aggregation_strategy": strategy
                }
                
                st.session_state.round_history.append(round_summary)
                save_round_summary(round_num, round_summary)
                
                progress_bar.progress(1.0)
                st.success(f"✓ Round {round_num} Complete!")
        
        with col2:
            if st.button("⏸️ Pause", key="pause", use_container_width=True):
                st.session_state.server_running = False
    else:
        st.info("Start server to simulate rounds")
    
    st.divider()
    
    # Round History
    st.subheader("Round History")
    
    if st.session_state.round_history:
        history_df = pd.DataFrame([
            {
                "Round": h["round"],
                "Timestamp": h["timestamp"],
                "Clients": h["clients_trained"],
                "Avg Accuracy": f"{h['global_accuracy']*100:.2f}%",
                "Avg Loss": f"{h['global_loss']:.4f}"
            }
            for h in st.session_state.round_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Plot accuracy and loss over rounds
        st.subheader("Metrics Over Rounds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_data = pd.DataFrame({
                "Round": [h["round"] for h in st.session_state.round_history],
                "Average Accuracy": [h["global_accuracy"]*100 for h in st.session_state.round_history]
            })
            st.line_chart(accuracy_data.set_index("Round"))
        
        with col2:
            loss_data = pd.DataFrame({
                "Round": [h["round"] for h in st.session_state.round_history],
                "Average Loss": [h["global_loss"] for h in st.session_state.round_history]
            })
            st.line_chart(loss_data.set_index("Round"))
    else:
        st.info("No rounds completed yet")

# ============================================================================
# TAB 4: AGGREGATION
# ============================================================================
with tab4:
    st.header("Model Aggregation")
    
    st.subheader("Aggregation Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Strategy Details")
        if strategy == "FedAvg":
            st.write("""
            **Federated Averaging (FedAvg)**
            
            - Weighted average of client model weights
            - Weights based on number of samples per client
            - Formula: w<sub>t+1</sub> = Σ(n<sub>k</sub>/n) * w<sub>k</sub>
            - Best for IID (balanced) data distributions
            """)
        else:
            st.write("""
            **FedProx**
            
            - Proximal term to stabilize convergence
            - Handles Non-IID (unbalanced) data distributions
            - Formula: includes μ control term
            - More robust to client drift
            """)
    
    with col2:
        st.write("### Current Settings")
        st.metric("Strategy", strategy)
        st.metric("Min Clients", MIN_FIT_CLIENTS)
        if strategy == "FedProx":
            st.metric("μ (mu)", mu)
    
    st.divider()
    
    # Simulated Aggregation
    st.subheader("Manual Aggregation (Testing)")
    
    if st.button("🔄 Trigger Manual Aggregation", key="manual_agg", use_container_width=True):
        if st.session_state.round_history:
            # Load latest round metrics
            latest_round = st.session_state.round_history[-1]
            
            st.info(f"Aggregating models from Round {latest_round['round']}...")
            
            # Display aggregation info
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Clients Contributing:**")
                for hospital, metrics in latest_round["client_metrics"].items():
                    st.write(f"- {hospital}: {metrics['samples_trained']} samples")
            
            with col2:
                st.write("**Aggregation Result:**")
                st.write(f"- Global Accuracy: {latest_round['global_accuracy']*100:.2f}%")
                st.write(f"- Global Loss: {latest_round['global_loss']:.4f}")

# ============================================================================
# TAB 5: ANALYTICS
# ============================================================================
with tab5:
    st.header("Training Analytics")
    
    if st.session_state.round_history:
        st.subheader("Global Model Performance")
        
        # Performance metrics
        latest = st.session_state.round_history[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Latest Round", latest["round"])
        with col2:
            st.metric("Avg Accuracy", f"{latest['global_accuracy']*100:.2f}%")
        with col3:
            st.metric("Avg Loss", f"{latest['global_loss']:.4f}")
        with col4:
            st.metric("Clients", latest["clients_trained"])
        
        st.divider()
        
        st.subheader("Detailed Round Analysis")
        
        # Select a round to analyze
        round_options = [h["round"] for h in st.session_state.round_history]
        selected_round_num = st.selectbox("Select Round", round_options)
        
        selected_round = next(h for h in st.session_state.round_history if h["round"] == selected_round_num)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Round Information**")
            st.write(f"- Round Number: {selected_round['round']}")
            st.write(f"- Timestamp: {selected_round['timestamp']}")
            st.write(f"- Strategy: {selected_round['aggregation_strategy']}")
        
        with col2:
            st.write("**Aggregated Metrics**")
            st.write(f"- Global Accuracy: {selected_round['global_accuracy']*100:.2f}%")
            st.write(f"- Global Loss: {selected_round['global_loss']:.4f}")
        
        st.divider()
        
        st.subheader("Client Performance in Round {selected_round_num}")
        
        client_metrics_df = pd.DataFrame([
            {
                "Hospital": hospital,
                "Accuracy": f"{m['accuracy']*100:.2f}%",
                "Loss": f"{m['loss']:.4f}",
                "Samples": m['samples_trained']
            }
            for hospital, m in selected_round["client_metrics"].items()
        ])
        
        st.dataframe(client_metrics_df, use_container_width=True)
        
        # Comparative visualization
        st.subheader("Comparative Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_df = pd.DataFrame({
                "Hospital": list(selected_round["client_metrics"].keys()),
                "Accuracy": [m['accuracy']*100 for m in selected_round["client_metrics"].values()]
            })
            st.bar_chart(accuracy_df.set_index("Hospital"))
        
        with col2:
            loss_df = pd.DataFrame({
                "Hospital": list(selected_round["client_metrics"].keys()),
                "Loss": [m['loss'] for m in selected_round["client_metrics"].values()]
            })
            st.bar_chart(loss_df.set_index("Hospital"))
    
    else:
        st.info("No training data available. Run federated rounds to see analytics.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("Federated Learning Server UI | Privacy-Preserving Healthcare AI")
