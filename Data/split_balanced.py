import json
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration for SISA-style slicing
NUM_SLICES = 5
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "Initial data" / "heart_disease_data.csv"
OUTPUT_DIR = BASE_DIR / "Balanced_split_data"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH, header=None)

    # Keep a stable patient identifier for unlearning bookkeeping.
    df["PatientID"] = df.index

    print("Creating sliced balanced distributions for SISA unlearning...")

    # Shuffle once to create balanced random partitions across hospitals.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # np.array_split keeps all rows (including remainders) distributed.
    hospital_frames = {
        name: df.iloc[idx].reset_index(drop=True)
        for name, idx in zip(HOSPITALS, np.array_split(df.index, len(HOSPITALS)))
    }

    for name, hospital_df in hospital_frames.items():
        hospital_dir = OUTPUT_DIR / name
        hospital_dir.mkdir(parents=True, exist_ok=True)

        # Backward-compatible hospital CSV used by current training/client code.
        hospital_df.drop(columns=["PatientID"]).to_csv(
            OUTPUT_DIR / f"{name}.csv", index=False, header=False
        )

        # Slice each hospital dataset for SISA retraining during unlearning.
        slices = [
            hospital_df.iloc[idx].reset_index(drop=True)
            for idx in np.array_split(hospital_df.index, NUM_SLICES)
        ]

        patient_to_slice = {}
        slice_to_patients = {}

        for i, slice_df in enumerate(slices, start=1):
            slice_filename = f"{name}_slice_{i}.csv"

            # Keep training format unchanged (do not include PatientID in model input CSV).
            slice_df.drop(columns=["PatientID"]).to_csv(
                hospital_dir / slice_filename, index=False, header=False
            )

            patient_ids = [int(pid) for pid in slice_df["PatientID"].tolist()]
            slice_to_patients[f"slice_{i}"] = patient_ids
            for pid in patient_ids:
                patient_to_slice[str(pid)] = i

        metadata = {
            "hospital": name,
            "num_slices": NUM_SLICES,
            "patient_to_slice": patient_to_slice,
            "slice_to_patients": slice_to_patients,
        }

        with open(hospital_dir / f"{name}_slice_map.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Slicing complete. Data saved in {OUTPUT_DIR}/[Hospital_Name]/")


if __name__ == "__main__":
    main()