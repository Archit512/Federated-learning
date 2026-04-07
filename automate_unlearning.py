import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "Client"))

from unlearning_utils import manual_unlearn


def trigger_automatic_unlearn(patient_id, hospital_name, data_style="Balanced"):
    # First remove the patient from the target hospital slice data.
    manual_unlearn(
        patient_id=patient_id,
        hospital_name=hospital_name,
        data_style=data_style,
        project_root=project_root,
    )

    # Force a clean restart from the earliest global checkpoint by default.
    rollback_target = 0

    signal = {
        "rollback_to_round": rollback_target,
        "requesting_hospital": hospital_name,
        "deleted_patient": patient_id,
    }

    signal_path = project_root / "Server" / "rollback_signal.json"
    with open(signal_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2)

    print(f"[SIGNAL] Automatic rollback to Round {rollback_target} sent to Server.")


if __name__ == "__main__":
    trigger_automatic_unlearn(patient_id=105, hospital_name="Hospital_A")
