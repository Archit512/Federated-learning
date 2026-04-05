from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "streamlit_predictions.db"
MODEL_PATH = PROJECT_ROOT / "Centralized" / "global_model.pth"

HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]
FEATURE_NAMES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "Diabetes",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
GRAPH_FILENAMES = [
    "local_hospital_accuracy.png",
    "centralized_server_accuracy.png",
    "local_vs_centralized_comparison.png",
]


class HeartRiskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                hospital TEXT NOT NULL,
                features_json TEXT NOT NULL,
                predicted_class INTEGER NOT NULL,
                probability REAL NOT NULL
            )
            """
        )
        conn.commit()


def save_prediction(hospital: str, features: list[float], predicted_class: int, probability: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (created_at, hospital, features_json, predicted_class, probability)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hospital,
                json.dumps(features),
                int(predicted_class),
                float(probability),
            ),
        )
        conn.commit()


def fetch_recent_predictions(limit: int = 50) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, created_at, hospital, predicted_class, probability
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    return df


@st.cache_resource
def load_model() -> HeartRiskModel:
    model = HeartRiskModel()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_prediction(model: HeartRiskModel, features: list[float]) -> tuple[int, float]:
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        logit = model(x)
        probability = torch.sigmoid(logit).item()

    predicted_class = 1 if probability >= 0.5 else 0
    return predicted_class, probability


def hospital_graph_paths(hospital: str) -> list[Path]:
    results_dir = PROJECT_ROOT / "Client" / hospital / "results"
    return [results_dir / graph_name for graph_name in GRAPH_FILENAMES]


def render_header() -> None:
    st.set_page_config(page_title="Federated Healthcare Demo", layout="wide")
    st.title("Federated Learning Healthcare Prediction UI")
    st.caption(
        "Select a hospital, enter all 21 features, run prediction, store the record, and review training graphs."
    )


def render_model_status() -> None:
    if MODEL_PATH.exists():
        st.success(f"Centralized model found: {MODEL_PATH}")
    else:
        st.error(
            "Centralized model is missing. Run centralized training first to create Centralized/global_model.pth"
        )


def feature_inputs() -> list[float]:
    st.subheader("Input Features (21)")
    values: list[float] = []

    cols = st.columns(3)
    for idx, feature_name in enumerate(FEATURE_NAMES):
        col = cols[idx % 3]
        value = col.number_input(
            label=feature_name,
            value=0.0,
            step=0.1,
            format="%.4f",
            key=f"feature_{idx + 1}",
        )
        values.append(float(value))

    return values


def render_graphs(hospital: str) -> None:
    st.subheader(f"Training Graphs: {hospital}")
    paths = hospital_graph_paths(hospital)

    found_any = False
    for path in paths:
        if path.exists():
            found_any = True
            st.image(str(path), caption=path.name, use_container_width=True)
        else:
            st.info(f"Graph not found: {path}")

    if not found_any:
        st.warning(
            "No graphs available yet for this hospital. Run federated training once so client results are generated."
        )


def render_recent_predictions() -> None:
    st.subheader("Recent Stored Predictions")
    df = fetch_recent_predictions(limit=50)
    if df.empty:
        st.info("No predictions have been stored yet.")
        return

    df["probability"] = (df["probability"] * 100.0).round(2)
    df = df.rename(columns={"probability": "probability_percent"})
    st.dataframe(df, use_container_width=True)


def init_ui_state() -> None:
    if "prediction_ready" not in st.session_state:
        st.session_state["prediction_ready"] = False
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None


def main() -> None:
    init_db()
    init_ui_state()
    render_header()
    render_model_status()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Prediction Panel")
        hospital = st.selectbox("Select Hospital", HOSPITALS)
        features = feature_inputs()

        if st.button("Run Prediction", type="primary"):
            if not MODEL_PATH.exists():
                st.error("Model not available. Please generate Centralized/global_model.pth first.")
            else:
                model = load_model()
                predicted_class, probability = run_prediction(model, features)
                save_prediction(hospital, features, predicted_class, probability)

                st.session_state["prediction_ready"] = True
                st.session_state["last_prediction"] = {
                    "hospital": hospital,
                    "predicted_class": predicted_class,
                    "probability": probability,
                }

    with right:
        if st.session_state["prediction_ready"] and st.session_state["last_prediction"] is not None:
            prediction = st.session_state["last_prediction"]
            label = "High Risk (Class 1)" if prediction["predicted_class"] == 1 else "Low Risk (Class 0)"
            st.subheader("Prediction Result")
            st.success(f"Prediction: {label}")
            st.metric("Class-1 Probability", f"{prediction['probability'] * 100:.2f}%")
            render_graphs(prediction["hospital"])

    st.divider()
    render_recent_predictions()


if __name__ == "__main__":
    main()
