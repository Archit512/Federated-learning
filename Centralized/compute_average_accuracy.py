"""
Runs centralized-model evaluation on all four hospitals and prints a summary.

Usage (from project root):
    python Centralized/compute_average_accuracy.py

Assumes:
  - Centralized/global_model.pth exists (run server /train first)
  - Data/Balanced_split_data/Hospital_X.csv files exist
"""

import sys
import os

# Allow importing evaluate_centralized from Client/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Client"))

from evaluate_centralized import evaluate_hospital

HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C", "Hospital_D"]

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "Data", "Balanced_split_data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "global_model.pth")


def main():
    print("=" * 50)
    print("  Centralized Model — Per-Hospital Evaluation")
    print("=" * 50)

    accuracies = []
    for hospital in HOSPITALS:
        csv_path = os.path.join(DATA_DIR, f"{hospital}.csv")
        acc = evaluate_hospital(MODEL_PATH, csv_path)
        print(f"  {hospital} accuracy = {acc:.4f}  ({acc*100:.2f}%)")
        accuracies.append(acc)

    avg = sum(accuracies) / len(accuracies)

    print("=" * 50)
    print(f"  Average Centralized Accuracy : {avg*100:.2f}%")
    print("=" * 50)
    print()
    print("  Compare with your federated results:")
    print("    Federated Accuracy (FedAvg)  : __.__% ")
    print("    Federated Accuracy (FedProx) : __.__% ")
    print()
    print("  Fill in the federated numbers after running")
    print("  server_FedAvg.py / server_FedProx.py.")


if __name__ == "__main__":
    main()