"""HW4 - Member 2: figures for the Client 1 part of the report.

Run after local_train.py (--all) and simulate_fl.py. Reads results/member2_client1/ and writes
results/member2_client1/figures/:
    client1_local_learning_curve.png  local-only training of Client 1 (loss/accuracy per epoch)
    client1_fl_rounds.png             Client 1 during FL: local train loss, global model on its val split
    fl_global_test_rounds.png         global model on global_test per round vs. baselines
    comparison_macro_f1.png           macro-F1 on global_test: local-only vs centralised vs FL
    fl_confusion_matrix.png           final FL global model on global_test
"""
import json

import matplotlib
matplotlib.use("Agg")  # no window, just save files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fl_common as C

R = C.RESULTS_DIR
FIG = R / "figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", FIG / name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    local = json.loads((R / "client1" / "local_only_metrics.json").read_text())

    # 1) Local-only learning curve of Client 1 (proves local training works).
    h = pd.read_csv(R / "client1" / "local_only_history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(h.epoch, h.loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="local val")
    ax[0].set(title="Client 1 local-only: loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(h.epoch, h.accuracy, label="train"); ax[1].plot(h.epoch, h.val_accuracy, label="local val")
    ax[1].set(title="Client 1 local-only: accuracy", xlabel="epoch", ylim=(0, 1.02))
    for a in ax: a.legend(); a.grid(alpha=.3)
    save(fig, "client1_local_learning_curve.png")

    # 2) Client 1 inside the federation, round by round.
    c1 = pd.read_csv(R / "client1" / "fl_rounds.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(c1["round"], c1.train_loss, marker="o", ms=3, label="local training loss")
    ax[0].plot(c1["round"], c1.val_loss, marker="o", ms=3, label="global model, local val loss")
    ax[0].set(title="Client 1 during FL: loss", xlabel="FL round")
    ax[1].plot(c1["round"], c1.val_accuracy, marker="o", ms=3, label="global model accuracy")
    ax[1].plot(c1["round"], c1.val_f1_macro, marker="o", ms=3, label="global model macro-F1")
    ax[1].axhline(local["local_val"]["f1_macro"], ls="--", c="gray", label="local-only macro-F1")
    ax[1].set(title="Client 1 local val set", xlabel="FL round", ylim=(0, 1.02))
    for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
    save(fig, "client1_fl_rounds.png")

    # 3) Global model on global_test per round, with the baselines as horizontal lines.
    s = pd.read_csv(R / "fl_simulation" / "server_rounds.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(s["round"], s.global_test_accuracy, marker="o", ms=3, label="FL global accuracy")
    ax.plot(s["round"], s.global_test_f1_macro, marker="o", ms=3, label="FL global macro-F1")
    base = pd.read_csv(R / "baselines.csv") if (R / "baselines.csv").exists() else None
    if base is not None:
        for _, b in base.iterrows():
            style = "-." if "centralised" in b.model else ":"
            ax.axhline(b.test_f1_macro, ls=style, lw=1, c="gray")
            ax.text(s["round"].max(), b.test_f1_macro, " " + b.model, va="center", fontsize=7)
    ax.set_xticks(range(0, int(s["round"].max()) + 1, 2))
    ax.set_xlim(-0.5, s["round"].max() + 4.5)       # room for the baseline labels
    ax.set(title="Global model on global test set (n=%d)" % len(C.load_csv("global_test.csv")[1]),
           xlabel="FL round (0 = initial weights)", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    save(fig, "fl_global_test_rounds.png")

    # 4) Bar chart: macro-F1 on global_test.
    final = json.loads((R / "fl_simulation" / "final_metrics.json").read_text())
    if base is not None:
        names = list(base.model) + ["federated (FedAvg)"]
        vals = list(base.test_f1_macro) + [final["global_test"]["f1_macro"]]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        colors = ["#9aa5b1"] * (len(names) - 2) + ["#5b7fa6", "#d9822b"]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(range(len(names)), [n.replace(" ", "\n", 1) for n in names], fontsize=8)
        ax.set(ylabel="macro-F1 on global test", ylim=(0, 1), title="Local-only vs centralised vs federated")
        ax.grid(axis="y", alpha=.3)
        save(fig, "comparison_macro_f1.png")

    # 5) Confusion matrix of the final FL global model.
    cm = np.array(final["global_test_confusion"])
    fam = final["families"]
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(fam)):
        for j in range(len(fam)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(fam)), fam, rotation=30, ha="right")
    ax.set_yticks(range(len(fam)), fam)
    ax.set(xlabel="predicted", ylabel="true", title="FL global model (global test)")
    save(fig, "fl_confusion_matrix.png")


if __name__ == "__main__":
    main()
