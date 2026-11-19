"""HW4 - Federated Learning malware classification: SHARED definitions.

Everything in this file must be IDENTICAL on Client 1, Client 2, Client 3 and the server:
  * the model architecture (FedAvg averages the weights layer by layer, so every
    participant must have exactly the same layers, shapes and layer order),
  * the way a client dataset is loaded (same 1,040 feature columns, same label map),
  * the local train/validation split rule and the training hyper-parameters.

Input data comes from Member 1 (data/clients/*.csv + feature_columns.json). The data is
already cleaned, scaled and normalised, so NO extra preprocessing is done here.
"""
import json
import os
from pathlib import Path

# Keep TensorFlow quiet (must be set before TensorFlow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------------------
# Paths. Default layout (same as the Cyber_HW4 repo):
#   <root>/data/clients/client1.csv ...      (Member 1)
#   <root>/member2_client1/fl_common.py      (this file)
#   <root>/results/member2_client1/          (everything this folder writes)
# Override with the environment variable HW4_DATA_DIR if your data lives elsewhere.
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = Path(os.environ.get("HW4_DATA_DIR", ROOT / "data" / "clients"))
RESULTS_DIR = ROOT / "results" / "member2_client1"  # own folder: no clash with teammates

# --------------------------------------------------------------------------------------
# Shared configuration (agreed by the whole team).
# --------------------------------------------------------------------------------------
SEED = 42              # same seed as the Midterm and Member 1's split
N_FEATURES = 1040      # 1,024 hashed API n-grams + 5 API stats + 11 network stats
N_CLASSES = 5          # agenttesla, amadey, asyncrat, formbook, lokibot
VAL_SIZE = 0.20        # each client keeps 20 % of ITS OWN data for local validation
LEARNING_RATE = 1e-3   # Adam
BATCH_SIZE = 16
LOCAL_EPOCHS = 5       # epochs per FL round on each client
NUM_ROUNDS = 20        # FL rounds -> 20 x 5 = 100 local epochs in total
USE_CLASS_WEIGHTS = True  # "balanced" weights; lokibot is the smallest family


def set_seed(seed=SEED):
    """Make Python, NumPy and TensorFlow deterministic (as far as TF allows on CPU)."""
    tf.keras.utils.set_random_seed(seed)


def build_model(seed=SEED):
    """THE shared model. Every client and the server must build it with this function.

    Architecture (MLP, 141,829 parameters):
        Input(1040) -> Dense(128, ReLU) -> Dropout(0.3)
                    -> Dense(64, ReLU)  -> Dropout(0.3)
                    -> Dense(5, softmax)
    Loss: sparse categorical cross-entropy (labels are integers 0..4).
    Optimizer: Adam, learning rate 1e-3.
    The layer names are fixed so the weight lists line up in the same order everywhere.
    """
    set_seed(seed)  # same seed -> same initial weights on every machine
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(N_FEATURES,), name="features"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
        tf.keras.layers.Dropout(0.3, name="dropout_2"),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="output"),
    ], name="hw4_malware_mlp")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def load_feature_info():
    """Return (feature column list, label_map) written by Member 1."""
    info = json.loads((DATA_DIR / "feature_columns.json").read_text())
    feats = info["feature_columns"]
    assert len(feats) == N_FEATURES, "expected %d features, got %d" % (N_FEATURES, len(feats))
    return feats, info["label_map"]


def load_csv(name):
    """Load one of Member 1's files and return (X, y).

    Only the 1,040 feature columns are used as input. sample_id, family and
    label_agreement are metadata and are NEVER used as features.
    """
    feats, label_map = load_feature_info()
    d = pd.read_csv(DATA_DIR / name)
    X = d[feats].to_numpy(dtype="float32")
    y = d["label"].to_numpy(dtype="int64")
    assert np.isfinite(X).all(), "NaN/inf in " + name
    assert set(np.unique(y)) <= set(label_map.values()), "unknown label in " + name
    return X, y


def load_client(client_id):
    """Load client<k>.csv and split it into a local train and a local validation set.

    The split is stratified (every family appears in both parts) and seeded, so all
    clients do it the same way. The global test set is NOT touched here.
    Returns X_train, X_val, y_train, y_val (scikit-learn order).
    """
    X, y = load_csv("client%d.csv" % client_id)
    return train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=SEED)


def class_weights(y):
    """Balanced class weights computed from the client's OWN labels (nothing is shared)."""
    if not USE_CLASS_WEIGHTS:
        return None
    classes = np.arange(N_CLASSES)
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    out = {int(c): 1.0 for c in classes}
    out.update({int(c): float(v) for c, v in zip(present, w)})
    return out


def compute_metrics(model, X, y):
    """Accuracy, macro precision/recall/F1 and loss of `model` on (X, y)."""
    prob = model.predict(X, verbose=0)
    pred = prob.argmax(axis=1)
    loss = float(tf.keras.losses.sparse_categorical_crossentropy(y, prob).numpy().mean())
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def confusion(model, X, y):
    """Confusion matrix (rows = true family, columns = predicted family)."""
    pred = model.predict(X, verbose=0).argmax(axis=1)
    return confusion_matrix(y, pred, labels=list(range(N_CLASSES)))


def family_names():
    """Family names ordered by their integer label."""
    _, label_map = load_feature_info()
    return [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
