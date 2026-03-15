"""
Send a hospital's local CSV dataset to the centralized Flask server.

Usage:
    python Client/send_data.py Hospital_A
    python Client/send_data.py Hospital_B
    python Client/send_data.py Hospital_C
    python Client/send_data.py Hospital_D

The script looks for the CSV at:
    Data/Balanced_split_data/<hospital_name>.csv
"""

import sys
import os
import requests

SERVER_URL = "http://127.0.0.1:5000"   # Change to server IP if running remotely

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "Balanced_split_data")


def send_dataset(hospital_name: str):
    csv_path = os.path.join(DATA_DIR, f"{hospital_name}.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    print(f"[CLIENT] Uploading {hospital_name} dataset → {SERVER_URL}/upload ...")

    with open(csv_path, "rb") as f:
        response = requests.post(
            f"{SERVER_URL}/upload",
            files={"file": (f"{hospital_name}.csv", f, "text/csv")},
            data={"hospital_name": hospital_name},
        )

    if response.status_code == 200:
        print(f"[CLIENT] ✓ {response.json()['message']}")
    else:
        print(f"[CLIENT] ✗ Upload failed: {response.text}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python send_data.py <HospitalName>")
        print("Example: python send_data.py Hospital_A")
        sys.exit(1)

    send_dataset(sys.argv[1])