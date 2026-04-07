import argparse
import sys
from pathlib import Path

HOSPITAL_NAME = "Hospital_B"
CLIENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CLIENT_DIR.parent
sys.path.insert(0, str(CLIENT_DIR))

from unlearning_utils import manual_unlearn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Manual unlearning for {HOSPITAL_NAME}")
    parser.add_argument("--patient-id", type=int, required=True, help="PatientID to remove")
    parser.add_argument(
        "--data-style",
        type=str,
        default="Balanced",
        choices=["Balanced", "Unbalanced"],
        help="Dataset style root under Data/",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    manual_unlearn(
        patient_id=args.patient_id,
        hospital_name=HOSPITAL_NAME,
        data_style=args.data_style,
        project_root=PROJECT_ROOT,
    )
