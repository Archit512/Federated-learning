import os
import json
from typing import Dict, List, Optional, Tuple

import flwr
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(SERVER_DIR, "global_history")
ROLLBACK_SIGNAL_PATH = os.path.join(SERVER_DIR, "rollback_signal.json")
UNLEARN_SIGNAL_PATH = os.path.join(SERVER_DIR, "unlearn_signal.json")


class SisaStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_active = False
        self.reload_round: Optional[int] = None
        self._signal_consumed = False

    def reload_old_checkpoint(self, checkpoint_round: int) -> Optional[Parameters]:
        checkpoint_path = os.path.join(HISTORY_DIR, f"global_model_round_{checkpoint_round}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"[SISA] Requested checkpoint not found: {checkpoint_path}")
            return None

        params_list = torch.load(checkpoint_path, map_location="cpu")
        reloaded_parameters = ndarrays_to_parameters(params_list)
        self.initial_parameters = reloaded_parameters
        print(f"[SISA] Reloaded global checkpoint round {checkpoint_round} from {checkpoint_path}")
        return reloaded_parameters

    def apply_cleanup_signal(self, checkpoint_round: int) -> Optional[Parameters]:
        self.cleanup_active = True
        self.reload_round = checkpoint_round
        return self.reload_old_checkpoint(checkpoint_round)

    def _read_signal_round(self, signal_path: str) -> Optional[int]:
        if not os.path.exists(signal_path):
            return None

        try:
            with open(signal_path, "r", encoding="utf-8") as f:
                signal_data = json.load(f)
        except Exception as exc:
            print(f"[SISA] Could not read cleanup signal {signal_path}: {exc}")
            return None

        if not signal_data.get("cleanup", True) and "rollback_to_round" not in signal_data:
            return None

        checkpoint_round = signal_data.get("rollback_to_round", signal_data.get("reload_round", 1))
        return int(checkpoint_round)

    def _poll_cleanup_signal(self) -> None:
        if self._signal_consumed:
            return

        checkpoint_round = self._read_signal_round(ROLLBACK_SIGNAL_PATH)
        if checkpoint_round is None:
            checkpoint_round = self._read_signal_round(UNLEARN_SIGNAL_PATH)

        if checkpoint_round is None:
            return

        self.apply_cleanup_signal(checkpoint_round)
        self._signal_consumed = True

    def fit_config(self, server_round: int) -> Dict[str, Scalar]:
        self._poll_cleanup_signal()
        return {
            "server_round": server_round,
            "Cleanup": self.cleanup_active,
            "rollback_round": int(self.reload_round or 0),
            "RollbackRound": int(self.reload_round or 0),
            "CleanupRound": int(self.reload_round or 0),
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            params_list = parameters_to_ndarrays(aggregated_parameters)

            os.makedirs(HISTORY_DIR, exist_ok=True)

            save_path = os.path.join(HISTORY_DIR, f"global_model_round_{server_round}.pth")
            torch.save(params_list, save_path)
            print(f"--- Round {server_round} Global Model Archived ---")

        return aggregated_parameters, metrics


if __name__ == "__main__":
    strategy = SisaStrategy(
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )
    strategy.on_fit_config_fn = strategy.fit_config

    flwr.server.start_server(
        server_address="0.0.0.0:8089",
        config=flwr.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )