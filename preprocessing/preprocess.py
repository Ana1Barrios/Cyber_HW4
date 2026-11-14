#!/usr/bin/env python3
"""HW4 - Member 1: dataset + preprocessing for Federated Learning malware classification.

Inputs (from the Midterm, read-only; they live on the team Drive, NOT in this repo):
  - api_sequences.jsonl              cleaned Cuckoo API-call sequences (one JSON per sample)
  - network_dataset_labeled_270.csv  per-pcap network statistics
  - samples.csv (optional)           master metadata, only used for the `label_agreement` column

What this script does
  1. Merge API + network data on sample_id (hybrid = API features + network features).
  2. Clean: drop samples with an empty/invalid pcap, duplicates, missing labels; replace
     NaN/inf; check label consistency between both sources.
  3. Build ONE shared feature space (required so FedAvg can average same-shaped weights):
       - API n-grams (1-3) -> HashingVectorizer (stateless: no vocabulary has to be shared
         between clients) -> log1p -> L2 normalisation
       - 5 sequence statistics (log1p) and 11 network statistics (log1p)
       - numeric features standardised with mean/std computed ONLY on the training pool
         (the 3 client train sets), never on the global test set -> no leakage.
  4. Hold out a stratified GLOBAL TEST set (never seen by any client).
  5. Split the remaining samples into 3 IID, stratified client datasets.
  6. Write client1/2/3.csv, global_test.csv, feature_columns.json, preprocessing_params.json.

Usage:
  python preprocessing/preprocess.py --api-sequences PATH --network-csv PATH [--samples-csv PATH]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize

SEED = 42            # same seed as the Midterm
N_CLIENTS = 3
TEST_SIZE = 0.20     # global test set (stratified)
N_HASH = 1024        # size of the hashed API n-gram space
SEQ_STATS = ["n_calls", "n_collapsed", "n_procs", "n_unique_apis", "api_entropy"]
NET_STATS = ["packet_count", "total_bytes", "duration_seconds", "average_packet_bytes",
             "tcp_packets", "udp_packets", "dns_packets", "http_packets",
             "tls_packets", "icmp_packets", "other_packets"]
LOG_COLS = [c for c in SEQ_STATS if c != "api_entropy"] + NET_STATS  # heavy-tailed counts


def load_and_clean(api_path, net_path, samples_path, log):
    """Merge both sources and apply the cleaning rules. `log` collects what was removed."""
    api = pd.read_json(api_path, lines=True)
    net = pd.read_csv(net_path)
    log["api_rows"], log["network_rows"] = len(api), len(net)

    # Cleaning rule 1: invalid pcap (empty capture -> all-zero network features).
    bad = net[(net["pcap_status"] != "ok") | (net["packet_count"] <= 0)]
    log["dropped_invalid_pcap"] = bad["sample_id"].tolist()
    net = net.drop(bad.index)

    # Cleaning rule 2: duplicated samples (same sha256).
    for name, df in (("api", api), ("net", net)):
        dup = df["sha256"].duplicated(keep="first")
        log["duplicates_" + name] = int(dup.sum())
    api = api.drop_duplicates("sha256")
    net = net.drop_duplicates("sha256")

    df = api.merge(net[["sample_id", "family"] + NET_STATS], on="sample_id",
                   how="inner", suffixes=("", "_net"))
    log["only_in_one_source"] = sorted(set(api.sample_id) ^ set(net.sample_id))

    # Cleaning rule 3: both sources must agree on the label.
    assert (df["family"] == df["family_net"]).all(), "label mismatch between API and network"
    df = df.drop(columns="family_net")

    # Cleaning rule 4: missing / invalid values.
    df = df.dropna(subset=["family", "seq"])
    df = df[df["seq"].map(len) > 0]
    num = SEQ_STATS + NET_STATS
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    log["nan_values_before_imputation"] = int(df[num].isna().sum().sum())
    df[num] = df[num].fillna(df[num].median())      # none expected; safety net
    df[num] = df[num].clip(lower=0)                 # counts cannot be negative

    # label_agreement (AVClass vs MalwareBazaar) is metadata for the analysis, not a feature.
    if samples_path:                                # master table is the reference
        s = pd.read_csv(samples_path)[["sample_id", "label_agreement"]]
        df = df.drop(columns=["label_agreement"], errors="ignore").merge(s, on="sample_id", how="left")
    elif "label_agreement" not in df.columns:
        df["label_agreement"] = "unknown"
    return df.sort_values("sample_id").reset_index(drop=True)


def api_hash_features(seqs):
    """Stateless hashed 1-3-gram counts of the API sequence -> log1p -> L2 norm."""
    hv = HashingVectorizer(n_features=N_HASH, token_pattern=r"\S+", lowercase=False,
                           ngram_range=(1, 3), alternate_sign=False, norm=None)
    X = hv.transform([" ".join(s) for s in seqs])
    X.data = np.log1p(X.data)                       # sublinear term frequency
    return normalize(X, norm="l2").toarray().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-sequences", required=True)
    ap.add_argument("--network-csv", required=True)
    ap.add_argument("--samples-csv", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "clients"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {}

    df = load_and_clean(args.api_sequences, args.network_csv, args.samples_csv, log)
    families = sorted(df["family"].unique())
    label_map = {f: i for i, f in enumerate(families)}
    df["label"] = df["family"].map(label_map)

    # ---- split first, then fit the scaler on the training pool only (no leakage) ----
    idx = np.arange(len(df))
    pool_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=df["label"],
                                          random_state=SEED)
    skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=SEED)
    client_idx = [pool_idx[te] for _, te in skf.split(pool_idx, df["label"].iloc[pool_idx])]

    # ---- features ----
    X_api = api_hash_features(df["seq"])
    num = df[SEQ_STATS + NET_STATS].astype(float).copy()
    num[LOG_COLS] = np.log1p(num[LOG_COLS])
    mu, sd = num.iloc[pool_idx].mean(), num.iloc[pool_idx].std(ddof=0).replace(0, 1.0)
    num = ((num - mu) / sd).astype(np.float32)
    assert np.isfinite(num.to_numpy()).all() and np.isfinite(X_api).all()

    api_cols = ["api_h%04d" % i for i in range(N_HASH)]
    feat = pd.concat([pd.DataFrame(X_api, columns=api_cols),
                      num.rename(columns=lambda c: ("api_" if c in SEQ_STATS else "net_") + c)],
                     axis=1)
    meta = df[["sample_id", "family", "label", "label_agreement"]]
    full = pd.concat([meta, feat], axis=1)

    for k, ix in enumerate(client_idx, 1):
        full.iloc[ix].to_csv(out / ("client%d.csv" % k), index=False, float_format="%.6g")
    full.iloc[test_idx].to_csv(out / "global_test.csv", index=False, float_format="%.6g")

    feature_cols = list(feat.columns)
    (out / "feature_columns.json").write_text(json.dumps({
        "label_map": label_map, "n_features": len(feature_cols),
        "api_hashed_ngrams": api_cols[0] + " .. " + api_cols[-1],
        "api_statistics": ["api_" + c for c in SEQ_STATS],
        "network_statistics": ["net_" + c for c in NET_STATS],
        "feature_columns": feature_cols}, indent=1))
    (out / "preprocessing_params.json").write_text(json.dumps({
        "seed": SEED, "n_clients": N_CLIENTS, "test_size": TEST_SIZE, "n_hash": N_HASH,
        "ngram_range": [1, 3], "log1p_columns": LOG_COLS,
        "scaler_fitted_on": "union of the 3 client training sets (global test excluded)",
        "mean": mu.to_dict(), "std": sd.to_dict(), "cleaning_log": log}, indent=1))

    print("cleaning log:", json.dumps(log))
    print("samples: %d | families: %s | features: %d" % (len(df), families, len(feature_cols)))
    for name, ix in [("client1", client_idx[0]), ("client2", client_idx[1]),
                     ("client3", client_idx[2]), ("global_test", test_idx)]:
        print(name, len(ix), df.iloc[ix]["family"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
