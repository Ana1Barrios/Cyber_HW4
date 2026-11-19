# HW4 – Member 2: Federated Learning Client 1 (Flower + Keras)

Client 1 of the team's federated malware classifier. It loads `data/clients/client1.csv`
(Member 1), trains the shared Keras model locally and exchanges only model weights with the
FL server (FedAvg). Until Member 4's server is ready, `simulate_fl.py` runs a FedAvg server
and all 3 clients on one machine over real gRPC (localhost).

## For Members 3 and 4: what must be identical

FedAvg averages the weights layer by layer, so every client and the server must use the same
model, data loading and message format. The simplest way is to import `fl_common.py` from
this folder. Everything below is defined there.

| Item | Value |
|---|---|
| Versions | Python 3.11, `flwr==1.13.1`, `tensorflow==2.18.0` (see `requirements-fl.txt`) |
| Input | the 1,040 columns of `feature_columns.json`, label = `label` (0–4); no extra preprocessing |
| Never train on | `global_test.csv` |
| Local split | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`: train on 80 %, evaluate on 20 % |
| Model | `fl_common.build_model()`, see below |
| Training | Adam lr 1e-3, sparse categorical cross-entropy, batch 16, balanced class weights from the client's own labels |
| Strategy | FedAvg (weighted by `n_train`), all 3 clients every round, 20 rounds × 5 local epochs |
| Initial weights | `build_model().get_weights()` (seed 42) on the server |

```python
def build_model(seed=42):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1040,), name="features"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dropout(0.3, name="dropout_2"),
        tf.keras.layers.Dense(5, activation="softmax", name="output"),
    ], name="hw4_malware_mlp")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
```
The model has 141,829 parameters, and its weights are 6 arrays: (1040,128), (128,), (128,64),
(64,), (64,5), (5,).

**Client ↔ server messages** (each client is a `fl.client.NumPyClient`):
- the server sends `{"server_round", "local_epochs", "batch_size"}` as fit config (`on_fit_config_fn`)
- `fit` returns `(weights, n_train, {"train_loss", "train_accuracy", "fit_seconds"})`
- `evaluate` returns `(loss, n_val, {"accuracy", "precision_macro", "recall_macro", "f1_macro"})` on the local 20 %
- clients connect with `fl.client.start_client(server_address="<IP>:8080", client=..., insecure=True)`

`simulate_fl.py` is a working reference server (FedAvg + metric aggregation + per-round global
evaluation) that Member 4 can reuse. `client1.py --client-id K` shows a complete client for any K.

## Files

| File | What it does |
|---|---|
| `fl_common.py` | **shared**: model, data loading, local split, hyper-parameters, metrics |
| `client1.py` | the Flower client (`fit` = local training, `evaluate` = global model on the local val split). The simulation uses `--client-id 2/3` as stand-ins for Clients 2 and 3. |
| `local_train.py` | local-only training test of Client 1 (`--all`: baselines for clients 2, 3 and a centralised model) |
| `simulate_fl.py` | FedAvg server + 3 client processes on 127.0.0.1, 20 rounds × 5 local epochs |
| `plot_results.py` | figures for the report |
| `run_all.py` | runs the three steps above in order |
| `requirements-fl.txt` | pinned dependencies |

Layout (same as the `Cyber_HW4` repo):
```
data/clients/client1.csv ... feature_columns.json   (Member 1)
member2_client1/                                    (this folder)
results/member2_client1/                            (written by the scripts)
```

## Setup and run (Windows / miniconda)
```bat
conda create -n hw4fl python=3.11 -y
conda activate hw4fl
pip install -r member2_client1/requirements-fl.txt
cd member2_client1
python run_all.py --quick      :: smoke test: 2 FL rounds
python run_all.py              :: full run, about 1 minute on CPU, no GPU needed
```
To connect to a real server (Member 4), run `python client1.py --server <IP>:8080`.
If port 8080 is busy, run `python simulate_fl.py --port 8090`.

## Outputs (`results/member2_client1/`)
| File | Content |
|---|---|
| `client1/local_only_history.csv`, `local_only_metrics.json`, `local_only_model.keras` | local-only training test of Client 1 |
| `client1/fl_rounds.csv` | Client 1 per FL round: local train loss/acc, global model on its local val split |
| `client2/`, `client3/fl_rounds.csv` | the same for the simulated Clients 2 and 3 |
| `baselines.csv` | local-only Clients 1–3 and the centralised model, on the global test set |
| `fl_simulation/server_rounds.csv` | per round: global model on the global test set + client-averaged metrics |
| `fl_simulation/final_metrics.json`, `global_model.keras` | final FL global model |
| `figures/*.png` | learning curve, FL rounds, comparison, confusion matrix |

## Results (run on Tom's laptop, 16 Nov 2026; identical to the cloud test run)

| Model, evaluated on the global test set (n = 54) | Accuracy | Macro-F1 |
|---|---:|---:|
| Local-only Client 1 | 0.593 | 0.585 |
| Local-only Client 2 | 0.611 | 0.616 |
| Local-only Client 3 | 0.556 | 0.542 |
| Centralised (pooled) | 0.704 | 0.695 |
| **Federated (FedAvg, 20 rounds)** | **0.704** | **0.704** |

- **Local training works:** Client 1's training loss falls from 1.60 to 0.11 over 100 epochs.
- **Client 1's own validation split (15 samples):** the federated global model scores macro-F1 0.726, against 0.637 for the local-only model.
- **Federated vs. centralised:** the federated model matches the centralised model without any client sharing its raw data. The full run took 32 s.
- **Small test set:** with only 54 test samples, differences of a few points are noise.
