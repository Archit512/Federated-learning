import flwr
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Storage for metrics across rounds ────────────────────────────────────────
round_accuracies: List[float] = []       # average federated accuracy per round
client_accuracies: List[List[float]] = []  # per-client accuracies per round

CENTRALIZED_ACCURACY = None  # set this after running centralized training



class FedAvgWithLogging(FedAvg):

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated

        per_client = []
        for _, evaluate_res in results:
            acc = evaluate_res.metrics.get("accuracy", None)
            if acc is not None:
                per_client.append(float(acc))

        client_accuracies.append(per_client)

        if per_client:
            avg = sum(per_client) / len(per_client)
            round_accuracies.append(avg)

            print("\n" + "=" * 52)
            print(f"  Round {server_round:>2} / 10 — Evaluation Results")
            print("=" * 52)
            for i, acc in enumerate(per_client):
                bar = "█" * int(acc * 30)
                print(f"  Hospital {chr(65+i)}: {acc*100:6.2f}%  {bar}")
            print(f"  {'─'*46}")
            print(f"  Average  : {avg*100:6.2f}%")
            print("=" * 52)
        else:
            round_accuracies.append(0.0)

        return aggregated


def save_comparison_graph(federated_rounds: List[float], centralized_acc: Optional[float]):
    rounds = list(range(1, len(federated_rounds) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds, [a * 100 for a in federated_rounds],
            marker="o", linewidth=2.5, markersize=6,
            color="#1D9E75", label="Federated (FedAvg)")

    if centralized_acc is not None:
        ax.axhline(y=centralized_acc * 100, color="#D85A30",
                   linewidth=2, linestyle="--",
                   label=f"Centralized ({centralized_acc*100:.2f}%)")

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Federated vs Centralized — Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    path = "comparison_graph.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Graph saved → {path}")
    plt.close()


def print_ascii_graph(federated_rounds: List[float], centralized_acc: Optional[float]):
    print("\n" + "=" * 52)
    print("  Accuracy per Round (ASCII Graph)")
    print("=" * 52)

    max_val = max(federated_rounds + ([centralized_acc] if centralized_acc else []))
    bar_max = 40

    for i, acc in enumerate(federated_rounds):
        bar_len = int((acc / max_val) * bar_max) if max_val > 0 else 0
        bar = "█" * bar_len
        print(f"  R{i+1:>2}: {acc*100:5.2f}% |{bar}")

    if centralized_acc is not None:
        bar_len = int((centralized_acc / max_val) * bar_max) if max_val > 0 else 0
        bar = "▓" * bar_len
        print(f"  {'─'*46}")
        print(f"  CENT: {centralized_acc*100:5.2f}% |{bar}  (centralized)")

    print("=" * 52)

    
    if federated_rounds:
        final_fed = federated_rounds[-1]
        print(f"\n  Final Federated Accuracy  : {final_fed*100:.2f}%")
        if centralized_acc is not None:
            diff = (final_fed - centralized_acc) * 100
            sign = "+" if diff >= 0 else ""
            print(f"  Centralized Accuracy      : {centralized_acc*100:.2f}%")
            print(f"  Difference                : {sign}{diff:.2f}%")
    print()


if __name__ == "__main__":
    CENTRALIZED_ACCURACY = None

    strategy = FedAvgWithLogging(
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )

    print("\n" + "=" * 52)
    print("  Flower Federated Server — FedAvg")
    print("  Waiting for 4 clients on port 8080...")
    print("=" * 52 + "\n")

    flwr.server.start_server(
        server_address="[::]:8080",
        config=flwr.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    
    print_ascii_graph(round_accuracies, CENTRALIZED_ACCURACY)
    save_comparison_graph(round_accuracies, CENTRALIZED_ACCURACY)