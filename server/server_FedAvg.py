import os
from typing import Dict, List, Optional, Tuple

import flwr
import torch
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


class SisaStrategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            params_list = parameters_to_ndarrays(aggregated_parameters)

            history_dir = "global_history"
            os.makedirs(history_dir, exist_ok=True)

            save_path = os.path.join(history_dir, f"global_model_round_{server_round}.pth")
            torch.save(params_list, save_path)
            print(f"--- Round {server_round} Global Model Archived ---")

        return aggregated_parameters, metrics


if __name__ == "__main__":
    flwr.server.start_server(
        server_address="0.0.0.0:8089",
        config=flwr.server.ServerConfig(num_rounds=10),
        strategy=SisaStrategy(
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=4,
            on_fit_config_fn=lambda r: {"server_round": r},
        ),
    )