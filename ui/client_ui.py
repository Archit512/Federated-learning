"""
Streamlit UI for Federated Learning Client
Allows hospital staff to:
- View and manage their local data
- Train local models
- Connect to federated server
- Make predictions
- Monitor performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import (
    Model, load_data, create_data_loaders, get_device,
    save_model, load_model, evaluate_model, predict,
    get_data_statistics, save_metrics, load_metrics, format_metrics
)
from config import (
    HOSPITALS, FLOWERS_SERVER_CLIENT_ADDRESS, EPOCHS_PER_ROUND,
    BATCH_SIZE, LEARNING_RATE, PREDICTION_THRESHOLD, RISK_CATEGORIES
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Hospital Federated Learning Client",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 0rem 0rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "hospital_name" not in st.session_state:
    st.session_state.hospital_name = "Hospital_A"
if "model" not in st.session_state:
    st.session_state.model = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "round_metrics" not in st.session_state:
    st.session_state.round_metrics = []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_hospital_data_path(hospital_name):
    """Get path to hospital CSV data"""
    base_path = Path(__file__).parent.parent / "Data" / "Balanced_split_data"
    return base_path / f"{hospital_name}.csv"

def get_model_save_path(hospital_name):
    """Get path to save hospital model"""
    base_path = Path(__file__).parent.parent / "ui" / "models"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / f"{hospital_name}_model.pth"

def get_metrics_save_path(hospital_name):
    """Get path to save hospital metrics"""
    base_path = Path(__file__).parent.parent / "ui" / "metrics"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / f"{hospital_name}_metrics.json"

def train_local_model(model, train_loader, device, epochs=EPOCHS_PER_ROUND):
    """Train model locally for specified epochs"""
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    model.to(device)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
    
    return model, losses

def evaluate_local_model(model, test_loader, device):
    """Evaluate model on local test set"""
    return evaluate_model(model, test_loader, device)

# ============================================================================
# MAIN UI
# ============================================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext x='50' y='70' font-size='70' text-anchor='middle'%3E🏥%3C/text%3E%3C/svg%3E", width=60)
with col2:
    st.title("Hospital Federated Learning Client")
    st.caption("Privacy-Preserving Heart Disease Prediction System")

# ============================================================================
# SIDEBAR - HOSPITAL SELECTION AND CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    hospital_name = st.selectbox(
        "Select Hospital",
        options=HOSPITALS,
        index=0,
        key="hospital_select"
    )
    st.session_state.hospital_name = hospital_name
    
    st.divider()
    
    server_ip = st.text_input(
        "Flower Server Address",
        value=FLOWERS_SERVER_CLIENT_ADDRESS,
        help="IP:Port of the Flower server"
    )
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    data_path = get_hospital_data_path(hospital_name)
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, header=None)
            st.metric("Total Samples", len(df))
            st.metric("Features", df.shape[1] - 1)
        except:
            st.warning("Could not load data stats")

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🎯 Local Training", "🔮 Predictions", "📈 Performance"])

# ============================================================================
# TAB 1: DATA OVERVIEW
# ============================================================================
with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Load Data")
        if st.button("🔄 Load Hospital Data", key="load_data_btn"):
            try:
                data_path = get_hospital_data_path(hospital_name)
                X_train, X_test, y_train, y_test = load_data(str(data_path))
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.data_loaded = True
                st.success(f"✓ Data loaded for {hospital_name}")
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    with col2:
        if st.session_state.data_loaded:
            st.success("✓ Data Ready")
        else:
            st.info("ℹ️ Click 'Load Hospital Data' to proceed")
    
    st.divider()
    
    # Data Statistics
    if st.session_state.data_loaded:
        st.subheader("📈 Data Statistics")
        
        stats = get_data_statistics(
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Samples", stats["train_samples"])
        with col2:
            st.metric("Test Samples", stats["test_samples"])
        with col3:
            st.metric("Total Samples", stats["total_samples"])
        with col4:
            st.metric("Features", stats["feature_count"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Positive Ratio", f"{stats['train_positive_ratio']*100:.1f}%")
        with col2:
            st.metric("Test Positive Ratio", f"{stats['test_positive_ratio']*100:.1f}%")
        
        # Data Distribution
        st.subheader("Class Distribution")
        df_dist = pd.DataFrame({
            "Class": ["Healthy (0)", "Disease (1)"],
            "Train": [stats["negative_class_train"], stats["positive_class_train"]],
            "Test": [stats["negative_class_test"], stats["positive_class_test"]]
        })
        st.bar_chart(df_dist.set_index("Class"))
    else:
        st.info("Load data to see statistics")

# ============================================================================
# TAB 2: LOCAL TRAINING
# ============================================================================
with tab2:
    st.header("Local Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the 'Data Overview' tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Initialize Model")
            if st.button("🚀 Initialize New Model", key="init_model"):
                device = get_device()
                model = Model()
                model.to(device)
                st.session_state.model = model
                st.success(f"✓ Model initialized (Device: {device})")
        
        with col2:
            if st.session_state.model is not None:
                st.success("✓ Model Ready")
            else:
                st.info("Initialize model")
        
        st.divider()
        
        # Training Configuration
        st.subheader("Training Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=20, value=EPOCHS_PER_ROUND)
        with col2:
            batch_size = st.number_input("Batch Size", min_value=8, max_value=64, value=BATCH_SIZE)
        with col3:
            learning_rate = st.number_input("Learning Rate", value=LEARNING_RATE, format="%.6f")
        
        st.divider()
        
        # Train Local Model
        st.subheader("Train Local Model")
        if st.session_state.model is not None and st.button("🎯 Start Local Training", key="train_btn"):
            try:
                device = get_device()
                
                # Create data loaders
                train_loader, test_loader = create_data_loaders(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    batch_size=batch_size
                )
                
                # Progress bar
                progress_bar = st.progress(0)
                training_status = st.empty()
                loss_chart = st.empty()
                
                losses = []
                
                # Training loop
                for epoch in range(epochs):
                    st.session_state.model, epoch_losses = train_local_model(
                        st.session_state.model,
                        train_loader,
                        device,
                        epochs=1
                    )
                    losses.extend(epoch_losses)
                    
                    # Update progress
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    training_status.write(
                        f"Epoch {epoch+1}/{epochs} | Loss: {epoch_losses[0]:.4f}"
                    )
                    
                    # Update loss chart
                    loss_df = pd.DataFrame({"Loss": losses})
                    loss_chart.line_chart(loss_df)
                
                # Evaluate
                accuracy, loss = evaluate_local_model(
                    st.session_state.model,
                    test_loader,
                    device
                )
                
                # Save model
                model_path = get_model_save_path(hospital_name)
                save_model(st.session_state.model, str(model_path))
                
                # Save metrics
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size
                }
                metrics_path = get_metrics_save_path(hospital_name)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                st.session_state.round_metrics.append(metrics)
                
                st.divider()
                st.success("✓ Training Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Accuracy", format_metrics(accuracy, loss)["accuracy"])
                with col2:
                    st.metric("Test Loss", format_metrics(accuracy, loss)["loss"])
                
            except Exception as e:
                st.error(f"Error during training: {e}")
        else:
            if st.session_state.model is None:
                st.info("Initialize model first")

# ============================================================================
# TAB 3: PREDICTIONS
# ============================================================================
with tab3:
    st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Train a model first")
    else:
        st.subheader("Predictions on Local Test Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sample_size = st.number_input(
                "Number of samples to predict",
                min_value=1,
                max_value=len(st.session_state.X_test),
                value=10
            )
        
        with col2:
            threshold = st.slider("Prediction Threshold", 0.0, 1.0, PREDICTION_THRESHOLD, 0.05)
        
        if st.button("🔮 Generate Predictions", key="predict_btn"):
            try:
                device = get_device()
                st.session_state.model.to(device)
                
                X_sample = st.session_state.X_test[:sample_size]
                y_true = st.session_state.y_test[:sample_size]
                
                predictions, probabilities = predict(
                    st.session_state.model,
                    X_sample,
                    threshold=threshold,
                    device=device
                )
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    "Sample": range(1, sample_size + 1),
                    "True Label": y_true.flatten().astype(int),
                    "Prediction": predictions.flatten().astype(int),
                    "Confidence (%)": (probabilities.flatten() * 100).round(2),
                    "Risk Level": [
                        next(
                            (k for k, (low, high) in RISK_CATEGORIES.items()
                             if low <= prob <= high),
                            "Unknown"
                        ) for prob in probabilities.flatten()
                    ]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Accuracy
                correct = np.sum(predictions.flatten() == y_true.flatten())
                accuracy = correct / len(y_true)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predictions Correct", f"{correct}/{sample_size}")
                with col2:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col3:
                    st.metric("Threshold", f"{threshold:.2f}")
                
            except Exception as e:
                st.error(f"Error making predictions: {e}")

# ============================================================================
# TAB 4: PERFORMANCE METRICS
# ============================================================================
with tab4:
    st.header("Model Performance")
    
    model_path = get_model_save_path(hospital_name)
    metrics_path = get_metrics_save_path(hospital_name)
    
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                latest_metrics = json.load(f)
            
            st.subheader("Latest Training Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{latest_metrics['accuracy']*100:.2f}%")
            with col2:
                st.metric("Loss", f"{latest_metrics['loss']:.4f}")
            with col3:
                st.metric("Training Time", latest_metrics.get('timestamp', 'N/A'))
            
            st.divider()
            
            st.subheader("Training Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Epochs:** {latest_metrics.get('epochs', 'N/A')}")
            with col2:
                st.info(f"**Learning Rate:** {latest_metrics.get('learning_rate', 'N/A')}")
        
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    else:
        st.info("No training metrics available yet. Train a model first.")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("Federated Learning Client UI | Privacy-Preserving Healthcare AI")
