"""
Configuration constants for Federated Learning UI
"""

# Server Configuration
SERVER_ADDRESS = "127.0.0.1"
SERVER_PORT = 8080
FLOWER_SERVER_ADDRESS = f"[::]:8080"

# Client Configuration
FLOWERS_SERVER_CLIENT_ADDRESS = "127.0.0.1:8080"
CLIENT_PORTS = {
    "Hospital_A": 8081,
    "Hospital_B": 8082,
    "Hospital_C": 8083,
    "Hospital_D": 8084,
}

# Training Configuration
NUM_ROUNDS = 10
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Model Configuration
INPUT_FEATURES = 21
HIDDEN_UNITS = 16
OUTPUT_UNITS = 1

# Federated Learning Configuration
MIN_FIT_CLIENTS = 4
MIN_EVALUATE_CLIENTS = 4
MIN_AVAILABLE_CLIENTS = 4
FEDPROX_MU = 0.05

# UI Configuration
STREAMLIT_PAGE_ICON = "🏥"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR_STATE = "expanded"

# Data Configuration
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]
DATA_COLUMNS = 21
LABEL_COLUMN = 1

# Paths
BASE_DIR = "."
CLIENT_DIR = "Client"
SERVER_DIR = "server"
DATA_DIR = "Data"
UI_DIR = "ui"
MODELS_DIR = "models"
METRICS_DIR = "metrics"

# Disease Prediction Thresholds
PREDICTION_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.7

# Risk Categories
RISK_CATEGORIES = {
    "Low Risk": (0.0, 0.3),
    "Moderate Risk": (0.3, 0.7),
    "High Risk": (0.7, 1.0)
}
