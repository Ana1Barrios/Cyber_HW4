"""HW4 - Member 2: run a complete Federated Learning experiment on ONE machine.

This is a TEST HARNESS so Client 1 can be tested end to end before Member 4's real server
exists. It starts:
  * a Flower server with FedAvg on 127.0.0.1:<port> (in this process), and
  * 3 client processes (client1.py --client-id 1/2/3), each one training only on its own
    clientK.csv. Clients 2 and 3 are stand-ins for Members 3 and 4.
Everything talks over real gRPC on localhost, exactly like a distributed setup; to use
Member 4's server later only the --server address of client1.py changes.

    python simulate_fl.py                 # 20 rounds, 5 local epochs per round
    python simulate_fl.py --rounds 3      # quick smoke test

FedAvg: after each round the server replaces the global weights by the average of the
client weights, weighted by each client's number of training samples:
    w_global = sum_k (n_k / n) * w_k

After every round the server also evaluates the global model on global_test.csv
(EVALUATION ONLY, the number of rounds is fixed in advance and never tuned on it).
Member 5 does the official final evaluation.

Outputs (results/member2_client1/fl_simulation/):
    server_rounds.csv        per round: global-test metrics + client-averaged metrics
    final_metrics.json       final global model on global_test + confusion matrix
    global_model.keras       final global model
    and results/member2_client1/clientK/fl_rounds.csv written by every client
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

import fl_common as C

OUT = C.RESULTS_DIR / "fl_simulation"


def weighted_average(results):
    """Average client metrics weighted by their number of samples (for the server log)."""
    total = sum(n for n, _ in results)
    keys = {k for _, m in results for k in m if isinstance(m[k], (int, float))}
    return {k: sum(n * m[k] for n, m in results if k in m) / total for k in sorted(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=C.NUM_ROUNDS)
    ap.add_argument("--local-epochs", type=int, default=C.LOCAL_EPOCHS)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    address = "127.0.0.1:%d" % args.port

    X_te, y_te = C.load_csv("global_test.csv")
    global_model = C.build_model()               # same initial weights as every client
    history = {}                                 # round -> row for server_rounds.csv

    def evaluate_global(server_round, parameters, config):
        """Server-side evaluation of the aggregated global model (round 0 = initial)."""
        global_model.set_weights(parameters)
        m = C.compute_metrics(global_model, X_te, y_te)
        history.setdefault(server_round, {"round": server_round}).update(
            {"global_test_" + k: v for k, v in m.items()})
        print("[server] round %2d global test: acc=%.3f F1=%.3f loss=%.3f"
              % (server_round, m["accuracy"], m["f1_macro"], m["loss"]))
        return m["loss"], {k: v for k, v in m.items() if k != "loss"}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,          # all 3 clients every round
        min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3,
        initial_parameters=ndarrays_to_parameters(global_model.get_weights()),
        on_fit_config_fn=lambda r: {"server_round": r, "local_epochs": args.local_epochs,
                                    "batch_size": C.BATCH_SIZE},
        on_evaluate_config_fn=lambda r: {"server_round": r},
        evaluate_fn=evaluate_global,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start the 3 clients as separate processes; they retry until the server is up.
    here = Path(__file__).resolve().parent
    procs = [subprocess.Popen([sys.executable, str(here / "client1.py"),
                               "--server", address, "--client-id", str(k)], cwd=str(here))
             for k in (1, 2, 3)]

    t0 = time.time()
    try:
        hist = fl.server.start_server(server_address=address, strategy=strategy,
                                      config=fl.server.ServerConfig(num_rounds=args.rounds))
    finally:
        for p in procs:                                # make sure no client is left running
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    seconds = time.time() - t0

    # Client-averaged metrics collected by Flower (distributed = from the clients).
    for key, series in hist.metrics_distributed_fit.items():
        for r, v in series:
            history.setdefault(r, {"round": r})["clients_" + key] = v
    for key, series in hist.metrics_distributed.items():
        for r, v in series:
            history.setdefault(r, {"round": r})["clients_val_" + key] = v
    for r, v in hist.losses_distributed:
        history.setdefault(r, {"round": r})["clients_val_loss"] = v

    rows = [history[r] for r in sorted(history)]
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
    with open(OUT / "server_rounds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # global_model holds the weights of the last evaluate_fn call = final global model.
    final = C.compute_metrics(global_model, X_te, y_te)
    (OUT / "final_metrics.json").write_text(json.dumps({
        "rounds": args.rounds, "local_epochs": args.local_epochs, "clients": 3,
        "strategy": "FedAvg", "seconds": round(seconds, 1),
        "global_test": final, "n_global_test": int(len(y_te)),
        "global_test_confusion": C.confusion(global_model, X_te, y_te).tolist(),
        "families": C.family_names()}, indent=1))
    global_model.save(OUT / "global_model.keras")
    print("\nFL finished in %.0f s. Final global model on global_test: acc=%.3f macro-F1=%.3f"
          % (seconds, final["accuracy"], final["f1_macro"]))
    print("results in", OUT)
    if any(p.returncode not in (0, None) for p in procs):
        print("WARNING: a client exited with an error, check the log above")


if __name__ == "__main__":
    main()
