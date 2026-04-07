import json
from pathlib import Path

import numpy as np
import pandas as pd

NUM_SLICES = 5
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "Initial data" / "heart_disease_data.csv"
OUTPUT_DIR = BASE_DIR / "Unbalanced_split_data"


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(RAW_DATA_PATH, header=None)
	df["PatientID"] = df.index

	size = len(df) // 4
	print("Creating sliced non-IID distributions for SISA unlearning...")

	# Preserve existing non-IID assignment logic, then shuffle locally per hospital.
	df_sorted = df.sort_values(by=18, ascending=True)
	hospital_a = df_sorted.iloc[:size]

	remaining = df_sorted.iloc[size:]
	remaining = remaining.sort_values(by=[4, 0], ascending=False)
	hospital_b = remaining.iloc[:size]

	remaining = remaining.iloc[size:]
	remaining = remaining.sort_values(by=[6, 18], ascending=False)
	hospital_c = remaining.iloc[:size]
	hospital_d = remaining.iloc[size:]

	hospital_data = {
		"Hospital_A": hospital_a.sample(frac=1, random_state=42).reset_index(drop=True),
		"Hospital_B": hospital_b.sample(frac=1, random_state=42).reset_index(drop=True),
		"Hospital_C": hospital_c.sample(frac=1, random_state=42).reset_index(drop=True),
		"Hospital_D": hospital_d.sample(frac=1, random_state=42).reset_index(drop=True),
	}

	for name in HOSPITALS:
		h_df = hospital_data[name]
		h_dir = OUTPUT_DIR / name
		h_dir.mkdir(parents=True, exist_ok=True)

		# Backward-compatible hospital CSV used by current training/client code.
		h_df.drop(columns=["PatientID"]).to_csv(
			OUTPUT_DIR / f"{name}.csv", index=False, header=False
		)

		slices = [
			h_df.iloc[idx].reset_index(drop=True)
			for idx in np.array_split(h_df.index, NUM_SLICES)
		]

		patient_to_slice = {}
		slice_to_patients = {}

		for i, slice_df in enumerate(slices, start=1):
			slice_df.drop(columns=["PatientID"]).to_csv(
				h_dir / f"{name}_slice_{i}.csv", index=False, header=False
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

		with open(h_dir / f"{name}_slice_map.json", "w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2)

	print(f"Non-IID slicing complete. Data saved in {OUTPUT_DIR}/[Hospital_Name]/")


if __name__ == "__main__":
	main()