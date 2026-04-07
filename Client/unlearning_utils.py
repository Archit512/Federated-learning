import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def _resolve_data_dir(project_root: Path, hospital_name: str, data_style: str) -> Path:
    style_map = {
        "balanced": "Balanced_split_data",
        "unbalanced": "Unbalanced_split_data",
    }
    style_key = data_style.strip().lower()
    if style_key not in style_map:
        raise ValueError("data_style must be 'Balanced' or 'Unbalanced'")

    return project_root / "Data" / style_map[style_key] / hospital_name


def _load_slice_map(map_path: Path) -> Dict[str, Any]:
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_lookup(slice_map: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, list], bool]:
    if "patient_to_slice" in slice_map:
        patient_to_slice = {str(k): int(v) for k, v in slice_map["patient_to_slice"].items()}
        slice_to_patients = {
            str(k): [str(pid) for pid in v] for k, v in slice_map.get("slice_to_patients", {}).items()
        }
        return patient_to_slice, slice_to_patients, True

    patient_to_slice = {str(k): int(v) for k, v in slice_map.items()}
    return patient_to_slice, {}, False


def _backup_files(paths: list[Path], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in paths:
        shutil.copy2(path, backup_dir / f"{path.name}.{timestamp}.bak")


def manual_unlearn(
    patient_id: int,
    hospital_name: str,
    data_style: str = "Balanced",
    project_root: Path | None = None,
) -> None:
    project_root = project_root or Path(__file__).resolve().parents[1]
    patient_id_str = str(patient_id)

    data_dir = _resolve_data_dir(project_root, hospital_name, data_style)
    map_path = data_dir / f"{hospital_name}_slice_map.json"

    if not map_path.exists():
        print(f"[ERROR] Slice map not found: {map_path}")
        return

    slice_map = _load_slice_map(map_path)
    patient_to_slice, slice_to_patients, is_new_schema = _extract_lookup(slice_map)

    if patient_id_str not in patient_to_slice:
        print(f"[ERROR] Patient {patient_id} not found in {hospital_name} slice map.")
        return

    target_slice = int(patient_to_slice[patient_id_str])
    slice_key = f"slice_{target_slice}"
    slice_file = data_dir / f"{hospital_name}_slice_{target_slice}.csv"

    if not slice_file.exists():
        print(f"[ERROR] Target slice file not found: {slice_file}")
        return

    print(f"[FIND] Patient {patient_id} found in Slice {target_slice} of {hospital_name}.")

    backup_dir = project_root / "unlearning_backups" / hospital_name
    _backup_files([map_path, slice_file], backup_dir)

    df = pd.read_csv(slice_file, header=None)

    if is_new_schema:
        patients_in_slice = slice_to_patients.get(slice_key, [])
        if patient_id_str not in patients_in_slice:
            print(f"[ERROR] Patient {patient_id} not present in {slice_key} patient list.")
            return

        row_idx = patients_in_slice.index(patient_id_str)
        if row_idx >= len(df):
            print(
                f"[ERROR] Row index mismatch for patient {patient_id}: "
                f"metadata index {row_idx}, slice rows {len(df)}."
            )
            return

        df_cleaned = df.drop(index=row_idx).reset_index(drop=True)
        df_cleaned.to_csv(slice_file, index=False, header=False)

        patients_in_slice.pop(row_idx)
        slice_to_patients[slice_key] = patients_in_slice
        patient_to_slice.pop(patient_id_str, None)

        slice_map["patient_to_slice"] = patient_to_slice
        slice_map["slice_to_patients"] = slice_to_patients
    else:
        pid_col = df.columns[-1]
        df_cleaned = df[df[pid_col].astype(str) != patient_id_str]
        if len(df_cleaned) == len(df):
            print(
                "[ERROR] Legacy map format detected but no matching patient row found in CSV. "
                "No data was changed."
            )
            return

        df_cleaned.to_csv(slice_file, index=False, header=False)
        patient_to_slice.pop(patient_id_str, None)
        slice_map = patient_to_slice

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(slice_map, f, indent=2)

    print(f"[REMOVE] Patient {patient_id} purged from {slice_file.name}.")
    print(f"[UPDATE] Slice map updated: {map_path.name}")

    history_dir = project_root / "Server" / "global_history"
    checkpoint_dir = project_root / "Client" / hospital_name / "local_checkpoints"

    print("\n[ACTION REQUIRED]")
    print("1. SERVER rollback:")
    print(
        "   Restore a global checkpoint from before deletion "
        f"(expected under {history_dir})."
    )
    print("2. CLIENT checkpoint cleanup:")
    print(
        "   Delete checkpoints from impacted slice onward in "
        f"{checkpoint_dir}, then retrain rounds."
    )
    print("3. RESTART federated training so the cleaned slice is used.")
