"""HW4 - Member 2: LOCAL-ONLY training test for Client 1 (no federated learning).

Purpose: "test that local training works" before joining the FL rounds.
Client 1 trains the shared model on its own data for the same total number of epochs it
would do during FL (NUM_ROUNDS x LOCAL_EPOCHS) and we check that the loss goes down and
the model is better than chance (0.20 for 5 balanced classes).

    python local_train.py            # Client 1 only
    python local_train.py --all      # also clients 2, 3 and a centralised (pooled) model,
                                     # as baselines to compare the FL result with

The global test set is only EVALUATED here (never trained on), so the numbers can be put
next to the FL global model and Member 1's reference numbers.

Outputs (results/member2_client1/):
    client1/local_only_history.csv   loss/accuracy per epoch
    client1/local_only_metrics.json  metrics on the local val split and on global_test
    client1/local_only_model.keras   the trained local-only model
    baselines.csv                    (--all) one row per baseline model
"""
import argparse
import json

import numpy as np
import pandas as pd

import fl_common as C

TOTAL_EPOCHS = C.NUM_ROUNDS * C.LOCAL_EPOCHS   # same training budget as in FL


def train_eval(name, X_tr, y_tr, X_va, y_va, X_te, y_te, out_dir=None):
    """Train a fresh shared model on (X_tr, y_tr), evaluate on the val and global test sets."""
    model = C.build_model()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=TOTAL_EPOCHS,
                     batch_size=C.BATCH_SIZE, class_weight=C.class_weights(y_tr),
                     shuffle=True, verbose=0)
    val = C.compute_metrics(model, X_va, y_va)
    test = C.compute_metrics(model, X_te, y_te)
    print("%-22s train=%3d | local val: acc=%.3f F1=%.3f | global test: acc=%.3f F1=%.3f"
          % (name, len(y_tr), val["accuracy"], val["f1_macro"], test["accuracy"], test["f1_macro"]))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        h = pd.DataFrame(hist.history)
        h.insert(0, "epoch", np.arange(1, len(h) + 1))
        h.to_csv(out_dir / "local_only_history.csv", index=False)
        (out_dir / "local_only_metrics.json").write_text(json.dumps({
            "model": name, "epochs": TOTAL_EPOCHS, "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)), "n_global_test": int(len(y_te)),
            "local_val": val, "global_test": test,
            "global_test_confusion": C.confusion(model, X_te, y_te).tolist(),
            "families": C.family_names()}, indent=1))
        model.save(out_dir / "local_only_model.keras")
        first, last = hist.history["loss"][0], hist.history["loss"][-1]
        print("  training loss %.3f -> %.3f (%s)"
              % (first, last, "OK, decreasing" if last < first else "WARNING: not decreasing"))
    return {"model": name, "n_train": len(y_tr),
            **{"val_" + k: v for k, v in val.items()},
            **{"test_" + k: v for k, v in test.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run baselines for clients 2, 3 and pooled")
    args = ap.parse_args()
    X_te, y_te = C.load_csv("global_test.csv")   # evaluation only

    rows = []
    X_tr, X_va, y_tr, y_va = C.load_client(1)
    rows.append(train_eval("local-only client 1", X_tr, y_tr, X_va, y_va, X_te, y_te,
                           out_dir=C.RESULTS_DIR / "client1"))
    if args.all:
        splits = {1: (X_tr, X_va, y_tr, y_va)}
        for k in (2, 3):
            splits[k] = C.load_client(k)
            a, b, c, d = splits[k]
            rows.append(train_eval("local-only client %d" % k, a, c, b, d, X_te, y_te))
        # Centralised upper bound: all client TRAIN splits pooled (what FL tries to approach).
        Xp = np.vstack([splits[k][0] for k in (1, 2, 3)])
        yp = np.concatenate([splits[k][2] for k in (1, 2, 3)])
        Xv = np.vstack([splits[k][1] for k in (1, 2, 3)])
        yv = np.concatenate([splits[k][3] for k in (1, 2, 3)])
        rows.append(train_eval("centralised (pooled)", Xp, yp, Xv, yv, X_te, y_te))
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "baselines.csv", index=False, float_format="%.4f")
        print("saved", C.RESULTS_DIR / "baselines.csv")


if __name__ == "__main__":
    main()
