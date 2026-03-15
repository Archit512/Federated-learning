import flwr

if __name__ == "__main__":
    # Start Flower server for 10 rounds of federated learning
    flwr.server.start_server(
        server_address="[::]:8080",
        config=flwr.server.ServerConfig(num_rounds=10),
        strategy=flwr.server.strategy.FedProx(
            min_fit_clients=4,
            min_evaluate_clients=4,
            min_available_clients=4,
            mu = 0.05
        ),
    )