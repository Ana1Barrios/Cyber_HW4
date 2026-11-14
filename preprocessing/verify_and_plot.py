#!/usr/bin/env python3
"""HW4 - Member 1: sanity checks on the client datasets + distribution figure + sanity baseline.

Checks: same columns in every file, no NaN/inf, no sample shared between files,
label/family consistency, then a quick CENTRALISED logistic regression (pool of the 3 clients
-> global test) as a reference number for Member 5 (NOT the FL result).
Usage: python preprocessing/verify_and_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "clients"
names = ["client1", "client2", "client3", "global_test"]
dfs = {n: pd.read_csv(D / (n + ".csv")) for n in names}
meta = json.loads((D / "feature_columns.json").read_text())
feats = meta["feature_columns"]

# ---- checks ----
cols = list(dfs["client1"].columns)
assert all(list(d.columns) == cols for d in dfs.values()), "column mismatch"
for n, d in dfs.items():
    assert np.isfinite(d[feats].to_numpy()).all(), n + " has NaN/inf"
    assert d["label"].map({v: k for k, v in meta["label_map"].items()}).eq(d["family"]).all()
ids = [set(d["sample_id"]) for d in dfs.values()]
assert sum(len(s) for s in ids) == len(set().union(*ids)), "a sample appears in two files"
print("OK: %d features, no NaN/inf, no overlap between files" % len(feats))

# ---- distribution table + figure ----
fams = sorted(meta["label_map"], key=meta["label_map"].get)
tab = pd.DataFrame({n: dfs[n]["family"].value_counts() for n in names}).reindex(fams)
tab["total"] = tab.sum(axis=1)
tab.loc["TOTAL"] = tab.sum()
agr = pd.DataFrame({n: dfs[n]["label_agreement"].value_counts() for n in names}).fillna(0).astype(int)
tab.to_csv(ROOT / "docs" / "client_distribution.csv")
print(tab.to_string()); print(agr.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
bottom = np.zeros(len(names))
colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
for f, c in zip(fams, colors):
    v = np.array([(dfs[n]["family"] == f).sum() for n in names])
    ax.bar(names, v, bottom=bottom, label=f, color=c)
    bottom += v
ax.set_ylabel("samples"); ax.set_title("Family distribution per client (IID, stratified)")
ax.set_ylim(0, 90); ax.legend(fontsize=8, ncol=3, loc="upper right"); fig.tight_layout()
fig.savefig(ROOT / "reports" / "figures" / "client_distribution.png", dpi=150)

# ---- centralised reference (sanity only) ----
pool = pd.concat([dfs[n] for n in names[:3]])
clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42)
clf.fit(pool[feats], pool["label"])
p = clf.predict(dfs["global_test"][feats]); y = dfs["global_test"]["label"]
print("centralised LR on pooled clients -> global_test: acc=%.3f macroF1=%.3f (n=%d)"
      % (accuracy_score(y, p), f1_score(y, p, average="macro"), len(y)))
for n in names[:3]:   # each client alone, to show local data is learnable but small
    c = LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42).fit(dfs[n][feats], dfs[n]["label"])
    q = c.predict(dfs["global_test"][feats])
    print("local-only %s: acc=%.3f macroF1=%.3f" % (n, accuracy_score(y, q), f1_score(y, q, average="macro")))
