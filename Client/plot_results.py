from pathlib import Path
from typing import List, Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _adaptive_ylim(values: List[float], padding: float = 0.03, min_span: float = 0.08) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0

    y_min = min(values)
    y_max = max(values)

    if (y_max - y_min) < min_span:
        center = (y_min + y_max) / 2
        half = min_span / 2
        y_min = center - half
        y_max = center + half

    y_min = max(0.0, y_min - padding)
    y_max = min(1.0, y_max + padding)

    if y_max <= y_min:
        return 0.0, 1.0
    return y_min, y_max


def save_local_hospital_plot(hospital_name: str, rounds: List[int], accuracies: List[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", linewidth=2)
    plt.title(f"{hospital_name} Local Federated Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.ylim(*_adaptive_ylim(accuracies))
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
    plt.ylim(*_adaptive_ylim([centralized_accuracy]))
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
    plt.ylim(*_adaptive_ylim(values))
    plt.tight_layout()
    plt.savefig(out_dir / "local_vs_centralized_comparison.png", dpi=150)
    plt.close()


def save_round_delta_plot(hospital_name: str, rounds: List[int], accuracies: List[float], out_dir: Path) -> None:
    if len(rounds) < 2 or len(accuracies) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    delta_rounds = rounds[1:]
    deltas = [accuracies[i] - accuracies[i - 1] for i in range(1, len(accuracies))]
    colors = ["tab:green" if d >= 0 else "tab:red" for d in deltas]

    plt.figure(figsize=(8, 5))
    plt.bar(delta_rounds, deltas, color=colors)
    plt.axhline(y=0.0, color="black", linewidth=1)
    plt.title(f"{hospital_name} Round-to-Round Accuracy Change")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy Delta")
    delta_min = min(deltas)
    delta_max = max(deltas)
    span = max(delta_max - delta_min, 0.02)
    pad = span * 0.25
    plt.ylim(delta_min - pad, delta_max + pad)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_delta_per_round.png", dpi=150)
    plt.close()


def save_accuracy_trend_plot(hospital_name: str, rounds: List[int], accuracies: List[float], out_dir: Path) -> None:
    if not rounds or not accuracies:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    series = pd.Series(accuracies)
    window = 3 if len(accuracies) >= 3 else len(accuracies)
    trend = series.rolling(window=window, min_periods=1).mean().tolist()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, marker="o", linewidth=1.5, alpha=0.5, label="Raw Accuracy")
    plt.plot(rounds, trend, marker="o", linewidth=2.5, color="tab:orange", label=f"{window}-Round Trend")
    plt.title(f"{hospital_name} Accuracy Trend")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.ylim(*_adaptive_ylim(accuracies + trend))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_trend_rolling.png", dpi=150)
    plt.close()


def save_local_vs_centralized_gap_plot(
    hospital_name: str,
    rounds: List[int],
    local_accuracies: List[float],
    centralized_accuracy: Optional[float],
    out_dir: Path,
) -> None:
    if centralized_accuracy is None or not rounds or not local_accuracies:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    gaps = [local_acc - centralized_accuracy for local_acc in local_accuracies]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, gaps, marker="o", linewidth=2, color="tab:purple")
    plt.axhline(y=0.0, color="black", linewidth=1)
    plt.title(f"{hospital_name} Gap vs Centralized Accuracy")
    plt.xlabel("Federated Round")
    plt.ylabel("Local - Centralized")
    gap_min = min(gaps)
    gap_max = max(gaps)
    span = max(gap_max - gap_min, 0.02)
    pad = span * 0.25
    plt.ylim(gap_min - pad, gap_max + pad)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "local_minus_centralized_gap.png", dpi=150)
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
    save_round_delta_plot(hospital_name, rounds, local_accuracies, out_dir)
    save_accuracy_trend_plot(hospital_name, rounds, local_accuracies, out_dir)
    save_local_vs_centralized_gap_plot(
        hospital_name,
        rounds,
        local_accuracies,
        centralized_accuracy,
        out_dir,
    )
