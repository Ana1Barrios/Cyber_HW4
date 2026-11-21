
"""HW4 - Member 4: Federated Learning Server (Flower + FedAvg).

Coordinates the three independent clients:
  Client 1: member2_client1/client1.py
  Client 2: member3_client2/client2.py
  Client 3: member4_client3/client3.py

Run from the repository root:
    python member4_server/server.py
    python member4_server/server.py --rounds 3
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import flwr as fl
from flwr.common import ndarrays_to_parameters

# Shared model, data utilities, and hyperparameters
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "member2_client1"))
import fl_common as C

OUT = ROOT / "results" / "member4_server"


def weighted_average(results):
    """Aggregate client metrics weighted by their sample counts."""
    total = sum(n for n, _ in results)
    if total == 0:
        return {}
    keys = {
        k for _, metrics in results
        for k, value in metrics.items()
        if isinstance(value, (int, float))
    }
    return {
        key: sum(n * metrics[key] for n, metrics in results if key in metrics) / total
        for key in sorted(keys)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=C.NUM_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=C.LOCAL_EPOCHS)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.rounds < 1 or args.local_epochs < 1:
        parser.error("--rounds and --local-epochs must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    address = f"127.0.0.1:{args.port}"
    history = {}

    # All clients use the same initial global model.
    global_model = C.build_model()

    def evaluate_global(server_round, parameters, config):
        """Evaluate aggregated weights on the held-out global test set."""
        X_test, y_test = C.load_csv("global_test.csv")
        global_model.set_weights(parameters)
        metrics = C.compute_metrics(global_model, X_test, y_test)

        history.setdefault(server_round, {"round": server_round}).update(
            {"global_test_" + key: value for key, value in metrics.items()}
        )
        print(
            f"[server] round {server_round}: "
            f"global accuracy={metrics['accuracy']:.3f}, "
            f"macro-F1={metrics['f1_macro']:.3f}"
        )
        return metrics["loss"], {
            key: value for key, value in metrics.items() if key != "loss"
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=ndarrays_to_parameters(global_model.get_weights()),
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": args.local_epochs,
            "batch_size": C.BATCH_SIZE,
        },
        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round
        },
        evaluate_fn=evaluate_global,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Launch the actual, independent client implementations.
    client_scripts = [
        ROOT / "member2_client1" / "client1.py",
        ROOT / "member3_client2" / "client2.py",
        ROOT / "member4_client3" / "client3.py",
    ]

    client_commands = [
        [sys.executable, str(client_scripts[0]), "--server", address, "--client-id", "1"],
        [sys.executable, str(client_scripts[1]), "--server", address],
        [sys.executable, str(client_scripts[2]), "--server", address],
    ]

    processes = []
    start = time.time()

    try:
        # Start clients before the server; they wait for the server to accept connections.
        for command in client_commands:
            processes.append(
                subprocess.Popen(command, cwd=str(ROOT))
            )

        history_obj = fl.server.start_server(
            server_address=address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
        )
    finally:
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    elapsed = time.time() - start

    # Save distributed metrics collected from the clients.
    for key, series in history_obj.metrics_distributed_fit.items():
        for round_number, value in series:
            history.setdefault(round_number, {"round": round_number})[
                "clients_" + key
            ] = value

    for key, series in history_obj.metrics_distributed.items():
        for round_number, value in series:
            history.setdefault(round_number, {"round": round_number})[
                "clients_val_" + key
            ] = value

    for round_number, value in history_obj.losses_distributed:
        history.setdefault(round_number, {"round": round_number})[
            "clients_val_loss"
        ] = value

    rows = [history[r] for r in sorted(history)]
    if rows:
        columns = sorted(
            {key for row in rows for key in row},
            key=lambda key: (key != "round", key),
        )
        with open(OUT / "server_rounds.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    # Save the final global model and its held-out test metrics.
    X_test, y_test = C.load_csv("global_test.csv")
    final_metrics = C.compute_metrics(global_model, X_test, y_test)

    summary = {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "clients": 3,
        "strategy": "FedAvg",
        "seconds": round(elapsed, 1),
        "global_test": final_metrics,
        "n_global_test": int(len(y_test)),
        "global_test_confusion": C.confusion(
            global_model, X_test, y_test
        ).tolist(),
        "families": C.family_names(),
    }

    (OUT / "final_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    global_model.save(OUT / "global_model.keras")

    print("\nFederated learning completed.")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Global test accuracy: {final_metrics['accuracy']:.3f}")
    print(f"Global test macro-F1: {final_metrics['f1_macro']:.3f}")
    print(f"Results saved to: {OUT}")

    failed = [
        process.returncode
        for process in processes
        if process.returncode not in (0, None)
    ]
    if failed:
        print("WARNING: One or more clients exited with an error:", failed)


if __name__ == "__main__":
    main()
