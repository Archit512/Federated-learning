from __future__ import annotations

import json
import re
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CHECKPOINT_PATTERN = re.compile(r"^round_(\d+)_slice_(\d+)\.pth$")
GLOBAL_HISTORY_PATTERN = re.compile(r"^global_model_round_(\d+)\.pth$")


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

    # Keep known plots first, then append any additional image outputs (for unlearning/custom runs).
    known_paths = [results_dir / graph_name for graph_name in GRAPH_FILENAMES]
    known_names = {path.name for path in known_paths}

    extra_paths: list[Path] = []
    if results_dir.exists():
        extra_paths = sorted(
            [
                path
                for path in results_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.name not in known_names
            ],
            key=lambda p: p.name.lower(),
        )

    return known_paths + extra_paths


def _resolve_server_history_dir() -> Path:
    uppercase = PROJECT_ROOT / "Server" / "global_history"
    lowercase = PROJECT_ROOT / "server" / "global_history"
    if uppercase.exists():
        return uppercase
    return lowercase


def _parse_checkpoint_rounds(checkpoint_dir: Path) -> tuple[dict[int, set[int]], int]:
    rounds_to_slices: dict[int, set[int]] = {}
    file_count = 0

    if not checkpoint_dir.exists():
        return rounds_to_slices, file_count

    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue

        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue

        round_num = int(match.group(1))
        slice_num = int(match.group(2))
        rounds_to_slices.setdefault(round_num, set()).add(slice_num)
        file_count += 1

    return rounds_to_slices, file_count


def _parse_global_rounds(history_dir: Path) -> set[int]:
    rounds: set[int] = set()
    if not history_dir.exists():
        return rounds

    for path in history_dir.iterdir():
        if not path.is_file():
            continue

        match = GLOBAL_HISTORY_PATTERN.match(path.name)
        if match is not None:
            rounds.add(int(match.group(1)))

    return rounds


def get_unlearning_status(hospital: str) -> dict[str, int | str | None]:
    checkpoint_dir = PROJECT_ROOT / "Client" / hospital / "local_checkpoints"
    history_dir = _resolve_server_history_dir()

    rounds_to_slices, checkpoint_count = _parse_checkpoint_rounds(checkpoint_dir)
    complete_rounds = sorted(round_num for round_num, slices in rounds_to_slices.items() if len(slices) >= 5)
    latest_retrain_round = complete_rounds[-1] if complete_rounds else None

    global_rounds = _parse_global_rounds(history_dir)
    local_rounds = set(rounds_to_slices.keys())
    inferred_rollback_candidates = sorted(global_rounds.intersection(local_rounds))
    inferred_rollback_round = (
        inferred_rollback_candidates[-1] if inferred_rollback_candidates else None
    )

    return {
        "checkpoint_count": checkpoint_count,
        "complete_round_count": len(complete_rounds),
        "latest_retrain_round": latest_retrain_round,
        "inferred_rollback_round": inferred_rollback_round,
        "checkpoint_dir": str(checkpoint_dir),
        "history_dir": str(history_dir),
    }


def render_unlearning_status(hospital: str) -> None:
    st.subheader(f"Unlearning Status: {hospital}")
    status = get_unlearning_status(hospital)

    c1, c2, c3 = st.columns(3)
    c1.metric("Checkpoint Files", status["checkpoint_count"])
    c2.metric(
        "Latest Retrain Round",
        status["latest_retrain_round"] if status["latest_retrain_round"] is not None else "N/A",
    )
    c3.metric(
        "Last Rollback Round (Inferred)",
        status["inferred_rollback_round"] if status["inferred_rollback_round"] is not None else "N/A",
    )

    st.caption(
        "Inference is based on overlap between local checkpoint rounds and server global-history rounds."
    )

    with st.expander("Status Details"):
        st.write(f"Complete retrain rounds (all 5 slices): {status['complete_round_count']}")
        st.write(f"Checkpoint directory: {status['checkpoint_dir']}")
        st.write(f"Server history directory: {status['history_dir']}")


def render_header() -> None:
    st.set_page_config(page_title="CardioShield", layout="wide")
    st.title("CardioShield: Federated Healthcare Prediction UI")
    st.caption(
        "Select a hospital, enter all 21 features, run prediction, store the record, and review training/unlearning graphs."
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
    st.subheader(f"Training and Unlearning Graphs: {hospital}")
    paths = hospital_graph_paths(hospital)
    available_paths = [path for path in paths if path.exists()]
    missing_core = [
        graph_name
        for graph_name in GRAPH_FILENAMES
        if not (PROJECT_ROOT / "Client" / hospital / "results" / graph_name).exists()
    ]

    if available_paths:
        st.caption(f"Displaying {len(available_paths)} graph image(s) from the selected hospital results folder.")
        for path in available_paths:
            st.image(str(path), caption=path.name, use_container_width=True)

        if missing_core:
            st.caption("Missing standard plots: " + ", ".join(missing_core))
    else:
        st.warning(
            "No graph images available yet for this hospital. Run federated training/unlearning so client result plots are generated."
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
        render_unlearning_status(hospital)
        render_graphs(hospital)

        if st.session_state["prediction_ready"] and st.session_state["last_prediction"] is not None:
            prediction = st.session_state["last_prediction"]
            label = "High Risk (Class 1)" if prediction["predicted_class"] == 1 else "Low Risk (Class 0)"
            st.subheader("Prediction Result")
            st.success(f"Prediction: {label}")
            st.metric("Class-1 Probability", f"{prediction['probability'] * 100:.2f}%")

    st.divider()
    render_recent_predictions()


if __name__ == "__main__":
    main()
