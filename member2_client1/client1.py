"""HW4 - Member 2: Federated Learning CLIENT 1 (Flower + Keras).

What Client 1 does in every FL round:
  1. receives the current GLOBAL model weights from the FL server,
  2. loads them into its local copy of the shared model (fl_common.build_model),
  3. trains locally on its OWN data only (data/clients/client1.csv, local train split),
  4. sends the updated weights + number of training samples back to the server
     (the server then averages the weights of all clients with FedAvg),
  5. when asked, evaluates the new global model on its local validation split.
Raw data never leaves the client: only model weights and a few metrics are sent.

Run against a server (e.g. Member 4's server, or simulate_fl.py on this machine):
    python client1.py --server 127.0.0.1:8080

The same class is reused by simulate_fl.py with --client-id 2 and 3, as stand-ins for
Members 3 and 4 until their own clients exist.
"""
import argparse
import csv
import time

import flwr as fl

import fl_common as C


class MalwareClient(fl.client.NumPyClient):
    """Flower client that trains the shared Keras model on one client's dataset."""

    def __init__(self, client_id=1):
        self.cid = client_id
        # Local data: train split for training, val split to evaluate the global model.
        X_tr, X_va, y_tr, y_va = C.load_client(client_id)
        self.X_train, self.y_train, self.X_val, self.y_val = X_tr, y_tr, X_va, y_va
        self.model = C.build_model()                 # identical architecture everywhere
        self.cw = C.class_weights(self.y_train)      # computed from local labels only
        self.round = 0                               # fallback round counter
        self.log = {}                                # round -> metrics, written to CSV
        self.log_path = C.RESULTS_DIR / ("client%d" % client_id) / "fl_rounds.csv"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print("[client %d] train=%d  val=%d  features=%d"
              % (client_id, len(self.y_train), len(self.y_val), self.X_train.shape[1]))

    # ---- Flower API -------------------------------------------------------------------
    def get_parameters(self, config):
        """Return the local model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """One round of LOCAL training starting from the global weights."""
        self.round = int(config.get("server_round", self.round + 1))
        epochs = int(config.get("local_epochs", C.LOCAL_EPOCHS))
        batch = int(config.get("batch_size", C.BATCH_SIZE))

        self.model.set_weights(parameters)           # start from the global model
        t0 = time.time()
        hist = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch,
                              class_weight=self.cw, shuffle=True, verbose=0)
        metrics = {"train_loss": float(hist.history["loss"][-1]),
                   "train_accuracy": float(hist.history["accuracy"][-1]),
                   "fit_seconds": round(time.time() - t0, 2)}
        self._record(metrics)
        print("[client %d] round %d fit: loss=%.4f acc=%.3f"
              % (self.cid, self.round, metrics["train_loss"], metrics["train_accuracy"]))
        # Updated weights + number of samples (FedAvg weights clients by this number).
        return self.model.get_weights(), len(self.y_train), metrics

    def evaluate(self, parameters, config):
        """Evaluate the (aggregated) global model on this client's local validation set."""
        self.model.set_weights(parameters)
        m = C.compute_metrics(self.model, self.X_val, self.y_val)
        self._record({"val_" + k: v for k, v in m.items()})
        print("[client %d] round %d eval: val_acc=%.3f val_f1=%.3f"
              % (self.cid, self.round, m["accuracy"], m["f1_macro"]))
        return m["loss"], len(self.y_val), {k: v for k, v in m.items() if k != "loss"}

    # ---- logging ----------------------------------------------------------------------
    def _record(self, metrics):
        """Store the metrics of the current round and rewrite the per-round CSV."""
        self.log.setdefault(self.round, {"round": self.round}).update(metrics)
        rows = [self.log[r] for r in sorted(self.log)]
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "round", k))
        with open(self.log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Federated Learning client (Flower + Keras)")
    ap.add_argument("--server", default="127.0.0.1:8080", help="FL server address host:port")
    ap.add_argument("--client-id", type=int, default=1, help="1 = Member 2's client")
    args = ap.parse_args()
    client = MalwareClient(args.client_id)
    # insecure=True: plain gRPC, fine for a local/lab network (no TLS certificates).
    fl.client.start_client(server_address=args.server, client=client.to_client(),
                           insecure=True, max_wait_time=120)


if __name__ == "__main__":
    main()
