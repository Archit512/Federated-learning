from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_local_hospital_plot(hospital_name: str, rounds: List[int], accuracies: List[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", linewidth=2)
    plt.title(f"{hospital_name} Local Federated Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "local_hospital_accuracy.png", dpi=150)
    plt.close()


def save_centralized_server_plot(hospital_name: str, centralized_accuracy: Optional[float], out_dir: Path) -> None:
    if centralized_accuracy is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.axhline(y=centralized_accuracy, color="tab:green", linewidth=2)
    plt.title(f"{hospital_name} Centralized Accuracy Reference")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "centralized_server_accuracy.png", dpi=150)
    plt.close()


def save_comparison_plot(
    hospital_name: str,
    local_accuracy: float,
    centralized_accuracy: Optional[float],
    out_dir: Path,
) -> None:
    if centralized_accuracy is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Local Hospital", "Centralized Server"]
    values = [local_accuracy, centralized_accuracy]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["tab:blue", "tab:green"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")

    plt.title(f"{hospital_name} Local vs Centralized Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "local_vs_centralized_comparison.png", dpi=150)
    plt.close()


def save_all_plots(
    hospital_name: str,
    rounds: List[int],
    local_accuracies: List[float],
    centralized_accuracy: Optional[float],
    out_dir: Path,
) -> None:
    save_local_hospital_plot(hospital_name, rounds, local_accuracies, out_dir)
    save_centralized_server_plot(hospital_name, centralized_accuracy, out_dir)
    save_comparison_plot(hospital_name, local_accuracies[-1], centralized_accuracy, out_dir)
