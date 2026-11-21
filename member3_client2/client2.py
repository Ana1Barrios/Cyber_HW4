"""HW4 - Member 3: Federated Learning CLIENT 2 (Flower + Keras).

In every FL round Client 2:
  1. receives the global model weights from the server,
  2. trains the shared model locally on its own data (data/clients/client2.csv),
  3. sends the updated weights and its number of training samples back,
  4. evaluates the aggregated global model on its local validation split.
Raw data never leaves this client; only weights and metrics are sent.

Run: python client2.py --server <IP>:8080
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import flwr as fl

# Use the team's shared definitions (model, data loading, hyper-parameters)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "member2_client1"))
import fl_common as C

CLIENT_ID = 2
RESULTS_DIR = ROOT / "results" / "member3_client2"   # our own results folder


class Client2(fl.client.NumPyClient):
    def __init__(self):
        # 80 % train / 20 % validation split of client2.csv (same rule for all clients)
        self.X_train, self.X_val, self.y_train, self.y_val = C.load_client(CLIENT_ID)
        self.model = C.build_model()                    # identical model on every client
        self.cw = C.class_weights(self.y_train)         # balanced weights from OUR labels only
        self.round = 0
        self.log = {}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = RESULTS_DIR / "client2_rounds.csv"
        print("[client 2] train=%d val=%d features=%d"
              % (len(self.y_train), len(self.y_val), self.X_train.shape[1]))

    def get_parameters(self, config):
        """Send the current local weights to the server."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """One round of local training, starting from the global weights."""
        self.round = int(config.get("server_round", self.round + 1))
        epochs = int(config.get("local_epochs", C.LOCAL_EPOCHS))
        batch = int(config.get("batch_size", C.BATCH_SIZE))
        self.model.set_weights(parameters)              # load the global model
        t0 = time.time()
        hist = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch,
                              class_weight=self.cw, shuffle=True, verbose=0)
        metrics = {"train_loss": float(hist.history["loss"][-1]),
                   "train_accuracy": float(hist.history["accuracy"][-1]),
                   "fit_seconds": round(time.time() - t0, 2)}
        self._record(metrics)
        print("[client 2] round %d fit: loss=%.4f acc=%.3f"
              % (self.round, metrics["train_loss"], metrics["train_accuracy"]))
        # updated weights + number of samples (FedAvg weights clients by this number)
        return self.model.get_weights(), len(self.y_train), metrics

    def evaluate(self, parameters, config):
        """Evaluate the aggregated global model on our local validation split."""
        self.model.set_weights(parameters)
        m = C.compute_metrics(self.model, self.X_val, self.y_val)
        self._record({"val_" + k: v for k, v in m.items()})
        print("[client 2] round %d eval: val_acc=%.3f val_f1=%.3f"
              % (self.round, m["accuracy"], m["f1_macro"]))
        return m["loss"], len(self.y_val), {k: v for k, v in m.items() if k != "loss"}

    def _record(self, metrics):
        """Save this round's metrics: one row per round in client2_rounds.csv."""
        self.log.setdefault(self.round, {"round": self.round}).update(metrics)
        rows = [self.log[r] for r in sorted(self.log)]
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
        with open(self.log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:8080", help="FL server host:port")
    args = ap.parse_args()
    fl.client.start_client(server_address=args.server, client=Client2().to_client(),
                           insecure=True, max_wait_time=120)