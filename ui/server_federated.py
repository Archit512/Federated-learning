"""
Refactored Federated Learning Server for Streamlit Integration
Seamlessly integrates with server_ui.py
"""

import flwr
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import Model, get_device

device = get_device()
print(f"Using device: {device}")


class FederatedLearningServer:
    """
    Central Federated Learning Server
    Coordinates training across hospital clients
    """
    
    def __init__(self, strategy="FedAvg", num_rounds=10, min_fit_clients=4, 
                 min_evaluate_clients=4, min_available_clients=4, fedprox_mu=0.05):
        """
        Initialize server
        
        Args:
            strategy: "FedAvg" or "FedProx"
            num_rounds: Number of federated learning rounds
            min_fit_clients: Minimum clients required for training
            min_evaluate_clients: Minimum clients required for evaluation
            min_available_clients: Minimum clients that must be available
            fedprox_mu: Proximal term for FedProx strategy
        """
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.fedprox_mu = fedprox_mu
        self.round_history = []
        
    def _create_strategy(self):
        """Create Flower aggregation strategy"""
        if self.strategy == "FedAvg":
            return flwr.server.strategy.FedAvg(
                min_fit_clients=self.min_fit_clients,
                min_evaluate_clients=self.min_evaluate_clients,
                min_available_clients=self.min_available_clients,
            )
        elif self.strategy == "FedProx":
            return flwr.server.strategy.FedProx(
                min_fit_clients=self.min_fit_clients,
                min_evaluate_clients=self.min_evaluate_clients,
                min_available_clients=self.min_available_clients,
                mu=self.fedprox_mu
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def start(self, server_address="[::]:8080", log_host: Optional[str] = None):
        """
        Start the Flower server
        
        Args:
            server_address: Server address to bind to
            log_host: Optional logging host
        """
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Server")
        print(f"{'='*60}")
        print(f"Strategy: {self.strategy}")
        print(f"Number of Rounds: {self.num_rounds}")
        print(f"Min Fit Clients: {self.min_fit_clients}")
        print(f"Min Evaluate Clients: {self.min_evaluate_clients}")
        print(f"Min Available Clients: {self.min_available_clients}")
        print(f"Server Address: {server_address}")
        
        if self.strategy == "FedProx":
            print(f"FedProx μ (mu): {self.fedprox_mu}")
        
        print(f"\nWaiting for clients to connect...")
        print(f"{'='*60}\n")
        
        # Create strategy
        strategy = self._create_strategy()
        
        # Start server
        try:
            flwr.server.start_server(
                server_address=server_address,
                config=flwr.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                grpc_max_send_msg_length=-1,
                grpc_max_receive_msg_length=-1
            )
        except Exception as e:
            print(f"\n[ERROR] Server failed: {e}")
            raise
    
    def get_round_info(self) -> Dict[str, Any]:
        """Get information about current round"""
        return {
            "strategy": self.strategy,
            "total_rounds": self.num_rounds,
            "round_history": self.round_history
        }


def start_server_simple(num_rounds=10, strategy="FedAvg", server_address="[::]:8080"):
    """
    Simple function to start Flower server
    
    Args:
        num_rounds: Number of federated learning rounds
        strategy: Aggregation strategy ("FedAvg" or "FedProx")
        server_address: Server address to bind to
    """
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Server (Simple Mode)")
    print(f"{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"Number of Rounds: {num_rounds}")
    print(f"Server Address: {server_address}")
    print(f"\nWaiting for 4 clients (Hospitals A, B, C, D) to connect...")
    print(f"{'='*60}\n")
    
    if strategy == "FedAvg":
        strategy_obj = flwr.server.strategy.FedAvg(
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=4,
        )
    elif strategy == "FedProx":
        strategy_obj = flwr.server.strategy.FedProx(
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=4,
            mu=0.05
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    flwr.server.start_server(
        server_address=server_address,
        config=flwr.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy_obj
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--strategy", default="FedAvg", choices=["FedAvg", "FedProx"],
                        help="Aggregation strategy")
    parser.add_argument("--server_address", default="[::]:8080", help="Server address to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    args = parser.parse_args()
    
    # Override server address with port if provided
    if args.port != 8080:
        args.server_address = f"[::]{args.port}"
    
    try:
        start_server_simple(
            num_rounds=args.num_rounds,
            strategy=args.strategy,
            server_address=args.server_address
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
