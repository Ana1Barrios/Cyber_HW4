"""HW4 - Member 5: evaluation of the final federated (FedAvg) global model.

What this script does
---------------------
1. Loads the FINAL global model produced by the FL server (Member 4,
   results/member4_server/global_model.keras). If member4_server/server.py has not been
   run on this machine, it falls back to Member 2's 3-client simulation.
2. Re-trains the comparison models with exactly the same model, data split and training
   budget (20 rounds x 5 local epochs = 100 epochs):
     * local-only Client 1, 2, 3  (each client alone, no collaboration)
     * centralised                (all client train splits pooled = upper bound for FL)
   They are re-trained here (not read from baselines.csv) because the evaluation needs
   their predictions for per-class metrics, confusion matrices and paired comparisons.
3. Evaluates every model on global_test.csv (54 samples, never used for training):
   accuracy, precision, recall, F1 (macro and weighted), per-family metrics, confusion
   matrix.
4. Quantifies uncertainty. 54 test samples is very small, so every headline number gets a
   95 % bootstrap confidence interval, and "FL vs. centralised" / "FL vs. local-only" is
   tested with a PAIRED bootstrap (same resampled test samples for both models).
5. Repeats the evaluation on the high-confidence subset (label_agreement = agree/partial,
   i.e. the Midterm label is confirmed by AVClass), to see how much label noise costs.
6. Writes tables (CSV/JSON/TXT) and figures (PNG) for the report and the slides.

Usage (from the repo root, Python 3.11 environment of member2_client1/requirements-fl.txt):
    python member5_evaluation/evaluate.py
    python member5_evaluation/evaluate.py --fl-model PATH/global_model.keras \
                                          --fl-rounds PATH/server_rounds.csv
    python member5_evaluation/evaluate.py --quick      # 200 bootstrap samples, fast check

--fl-model accepts a saved Keras model (.keras) or a NumPy weight list (.npz, the 6 arrays
of fl_common.build_model() in layer order, e.g. the FedAvg parameters saved by a server).

Outputs: results/member5_evaluation/ (tables) and results/member5_evaluation/figures/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The shared definitions (model, data loading, split, hyper-parameters) live in Member 2's
# fl_common.py. Importing them guarantees the evaluation uses the same setup as the FL run.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "member2_client1"))
import fl_common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # no display needed, figures are only written to files
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_fscore_support)

OUT = ROOT / "results" / "member5_evaluation"
FIG = OUT / "figures"
# Default input: the final run of Member 4's server (member4_server/server.py). If that run
# has not been done on this machine, fall back to Member 2's 3-client simulation.
_SERVER = ROOT / "results" / "member4_server"
_SIM = ROOT / "results" / "member2_client1" / "fl_simulation"
_RUN = _SERVER if (_SERVER / "global_model.keras").exists() else _SIM
DEFAULT_FL_MODEL = _RUN / "global_model.keras"
DEFAULT_FL_ROUNDS = _RUN / "server_rounds.csv"
HIGH_CONFIDENCE = ("agree", "partial")  # label confirmed (fully or partly) by AVClass

# --------------------------------------------------------------------------------------
# Figure style. One fixed colour per model (categorical palette, fixed order), so a model
# has the same colour in every figure. Text always stays dark grey, never the series colour.
# --------------------------------------------------------------------------------------
FL = "Federated (FedAvg)"
CEN = "Centralised"
LOCAL = ["Local-only client %d" % k for k in (1, 2, 3)]
MODELS = [FL, CEN] + LOCAL
COLORS = {FL: "#2a78d6", CEN: "#eb6834", LOCAL[0]: "#1baf7a",
          LOCAL[1]: "#eda100", LOCAL[2]: "#e87ba4"}
LOCAL_MEAN_COLOR = "#8a8984"  # neutral grey for the averaged local-only reference
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Sequential single-hue ramp (light -> dark blue) for the confusion matrices.
CM_CMAP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _rel(path):
    """Path relative to the repo root (no personal absolute paths in committed results)."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def load_fl_model(path):
    """Load the federated global model from a .keras file or an .npz weight list."""
    path = Path(path)
    if path.suffix == ".npz":
        model = C.build_model()
        data = np.load(path)
        model.set_weights([data[k] for k in data.files])  # arrays must be in layer order
        return model
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def train_baseline(X_tr, y_tr):
    """Train a fresh shared model with the same budget as one FL client (100 epochs)."""
    model = C.build_model()  # same architecture, same seed, same initial weights as FL
    model.fit(X_tr, y_tr, epochs=C.NUM_ROUNDS * C.LOCAL_EPOCHS, batch_size=C.BATCH_SIZE,
              class_weight=C.class_weights(y_tr), shuffle=True, verbose=0)
    return model


def baselines():
    """Local-only clients 1-3 and the centralised model, trained on the clients' 80 % splits."""
    splits = {k: C.load_client(k) for k in (1, 2, 3)}  # (X_tr, X_va, y_tr, y_va), seed 42
    models = {}
    for k in (1, 2, 3):
        X_tr, _, y_tr, _ = splits[k]
        models[LOCAL[k - 1]] = train_baseline(X_tr, y_tr)
        print("  trained %s (n_train=%d)" % (LOCAL[k - 1], len(y_tr)))
    X_pool = np.vstack([splits[k][0] for k in (1, 2, 3)])
    y_pool = np.concatenate([splits[k][2] for k in (1, 2, 3)])
    models[CEN] = train_baseline(X_pool, y_pool)
    print("  trained %s (n_train=%d)" % (CEN, len(y_pool)))
    return models


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def scores(y, pred):
    """The four headline metrics (macro = every family counts equally) + weighted F1."""
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(C.N_CLASSES),
                                                 average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r,
            "f1_macro": f, "f1_weighted": f1_score(y, pred, average="weighted",
                                                   labels=range(C.N_CLASSES), zero_division=0)}


def bootstrap(y, preds, n_boot, seed=C.SEED):
    """Percentile bootstrap over the test samples, paired across models.

    In every repetition the same 54 indices are drawn (with replacement) for ALL models, so
    the difference between two models is computed on identical test samples. Returns
    {model: {metric: array of n_boot values}}.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {m: {k: np.empty(n_boot) for k in ("accuracy", "precision_macro", "recall_macro",
                                             "f1_macro")} for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m, pred in preds.items():
            s = scores(y[idx], pred[idx])
            for k in out[m]:
                out[m][k][b] = s[k]
    return out


def ci(values):
    """95 % percentile confidence interval."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_difference(boot, a, b, metric="f1_macro"):
    """Bootstrap distribution of metric(a) - metric(b): mean, 95 % CI and P(a > b)."""
    d = boot[a][metric] - boot[b][metric]
    lo, hi = ci(d)
    return {"mean": float(d.mean()), "ci95": [lo, hi], "p_a_better": float((d > 0).mean()),
            "significant": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig_confusion(cm, families, path, title):
    """Confusion matrix of one model: counts (left) and row-normalised recall (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.45})
    norm = cm / cm.sum(axis=1, keepdims=True)
    for ax, mat, sub, fmt in ((axes[0], cm, "Counts", "%d"),
                              (axes[1], norm, "Row-normalised (recall per family)", "%.2f")):
        ax.grid(False)
        ax.imshow(mat, cmap=CM_CMAP, vmin=0, vmax=mat.max() if mat is cm else 1)
        for i in range(len(families)):
            for j in range(len(families)):
                v = mat[i, j]
                dark = v > (0.55 * (mat.max() if mat is cm else 1))
                ax.text(j, i, fmt % v, ha="center", va="center", fontsize=10,
                        color="#ffffff" if dark else TEXT)
        ax.set_xticks(range(len(families)), families, rotation=30, ha="right")
        ax.set_yticks(range(len(families)), families)
        ax.set_xlabel("Predicted family")
        if ax is axes[0]:
            ax.set_ylabel("True family")
        ax.set_title(sub, fontsize=10, fontweight="normal", color=TEXT2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontweight="bold", x=0.5)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion_grid(cms, families, path):
    """Confusion matrices of all five models side by side (same colour scale)."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(19, 4.2))
    vmax = max(cm.max() for cm in cms.values())
    for ax, m in zip(axes, MODELS):
        cm = cms[m]
        ax.grid(False)
        ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=vmax)
        for i in range(len(families)):
            for j in range(len(families)):
                ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=9,
                        color="#ffffff" if cm[i, j] > 0.55 * vmax else TEXT)
        ax.set_xticks(range(len(families)), families, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(families)), families if ax is axes[0] else [], fontsize=8)
        ax.set_title("%s\naccuracy %.2f" % (m, np.trace(cm) / cm.sum()), fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True family")
    fig.supxlabel("Predicted family", color=TEXT2, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_comparison(summary, path, n_boot):
    """Grouped bars: 4 metrics x 5 models, whiskers = 95 % bootstrap CI."""
    metrics = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
               ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)")]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.15
    x = np.arange(len(metrics))
    for i, m in enumerate(MODELS):
        row = summary[summary.model == m].iloc[0]
        vals = np.array([row[k] for k, _ in metrics])
        lo = vals - np.array([row[k + "_ci_low"] for k, _ in metrics])
        hi = np.array([row[k + "_ci_high"] for k, _ in metrics]) - vals
        xs = x + (i - 2) * width
        ax.bar(xs, vals, width - 0.02, color=COLORS[m], label=m, zorder=2)
        ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=TEXT2, elinewidth=1,
                    capsize=2, zorder=3)
        if m in (FL, CEN):  # direct labels only for the two models the question is about
            for xx, v in zip(xs, vals):
                ax.text(xx, 0.03, "%.2f" % v, ha="center", va="bottom", fontsize=8,
                        color="#ffffff", rotation=90, zorder=4)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on global test set (n = 54)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Federated vs. centralised vs. local-only training")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.text(0.01, -0.12, "Whiskers: 95 %% bootstrap confidence interval (%s resamples "
             "of the 54 test samples)." % format(n_boot, ","), fontsize=8, color=TEXT2)
    fig.savefig(path)
    plt.close(fig)


def fig_per_class(per_class, families, path):
    """Per-family F1: federated, centralised and the mean of the three local-only models."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    series = [(FL, COLORS[FL]), (CEN, COLORS[CEN]), ("Local-only (mean of 3)", LOCAL_MEAN_COLOR)]
    width = 0.26
    x = np.arange(len(families))
    for i, (name, color) in enumerate(series):
        if name.startswith("Local-only"):
            vals = np.mean([per_class[m]["f1"] for m in LOCAL], axis=0)
        else:
            vals = per_class[name]["f1"]
        ax.bar(x + (i - 1) * width, vals, width - 0.03, color=color, label=name, zorder=2)
    support = per_class[FL]["support"]
    ax.set_xticks(x, ["%s\n(n=%d)" % (f, s) for f, s in zip(families, support)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 on global test set")
    ax.grid(axis="x", visible=False)
    ax.set_title("F1 per malware family")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(rounds, summary, path):
    """Macro-F1 of the global model after every FedAvg round, with the two references."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    r = rounds["round"].to_numpy()
    f1 = rounds["global_test_f1_macro"].to_numpy()
    ax.plot(r, f1, color=COLORS[FL], linewidth=2, marker="o", markersize=4, zorder=3)
    ax.annotate("Federated  %.2f" % f1[-1], (r[-1], f1[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=TEXT)
    cen = float(summary.loc[summary.model == CEN, "f1_macro"].iloc[0])
    loc = float(summary.loc[summary.model.isin(LOCAL), "f1_macro"].mean())
    for val, color, lab, above in ((cen, COLORS[CEN], "Centralised  %.2f" % cen, True),
                                   (loc, LOCAL_MEAN_COLOR, "Local-only mean  %.2f" % loc, False)):
        ax.axhline(val, color=color, linewidth=2, linestyle="--", zorder=2)
        # label at the right end, above (centralised) or below (local) its line
        ax.text(r[-1], val + (0.015 if above else -0.015), lab, fontsize=9, color=TEXT,
                ha="right", va="bottom" if above else "top")
    ax.axhline(0.2, color=GRID, linewidth=1, zorder=1)
    ax.text(r[-1], 0.205, "chance (5 classes)", fontsize=8, color=TEXT2, ha="right", va="bottom")
    ax.set_xlim(0, r[-1] + 3)
    ax.set_xticks(r[::2])
    ax.set_ylim(0, 1)
    ax.set_xlabel("FedAvg round (round 0 = initial weights)")
    ax.set_ylabel("Macro-F1 on global test set")
    ax.set_title("Convergence of the federated global model")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fl-model", default=str(DEFAULT_FL_MODEL),
                    help="final FL global model (.keras or .npz weight list)")
    ap.add_argument("--fl-rounds", default=str(DEFAULT_FL_ROUNDS),
                    help="server_rounds.csv with global_test_f1_macro per round (optional)")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--quick", action="store_true", help="200 resamples instead of 2000")
    args = ap.parse_args()
    n_boot = 200 if args.quick else args.n_boot
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    families = C.family_names()

    # Global test set: only evaluated, never trained on. Metadata kept for the subset analysis.
    X_te, y_te = C.load_csv("global_test.csv")
    meta = pd.read_csv(C.DATA_DIR / "global_test.csv", usecols=["sample_id", "family",
                                                                 "label_agreement"])
    print("global test set: %d samples" % len(y_te))

    print("loading FL global model:", args.fl_model)
    models = {FL: load_fl_model(args.fl_model)}
    print("training comparison models (same model, split and 100-epoch budget):")
    models.update(baselines())

    # Predictions of every model on the same test samples.
    preds = {m: models[m].predict(X_te, verbose=0).argmax(axis=1) for m in MODELS}

    # ---- headline metrics with 95 % bootstrap CI -------------------------------------
    print("bootstrap (%d resamples) ..." % n_boot)
    boot = bootstrap(y_te, preds, n_boot)
    hc = meta["label_agreement"].isin(HIGH_CONFIDENCE).to_numpy()
    rows = []
    for m in MODELS:
        s = scores(y_te, preds[m])
        row = {"model": m, **s}
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            row[k + "_ci_low"], row[k + "_ci_high"] = ci(boot[m][k])
        hs = scores(y_te[hc], preds[m][hc])
        row["hc_accuracy"], row["hc_f1_macro"] = hs["accuracy"], hs["f1_macro"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "metrics_summary.csv", index=False, float_format="%.4f")

    # ---- per-family metrics and confusion matrices ------------------------------------
    per_class, cms, pc_rows = {}, {}, []
    for m in MODELS:
        p, r, f, sup = precision_recall_fscore_support(y_te, preds[m], labels=range(C.N_CLASSES),
                                                       zero_division=0)
        per_class[m] = {"precision": p, "recall": r, "f1": f, "support": sup}
        cms[m] = confusion_matrix(y_te, preds[m], labels=range(C.N_CLASSES))
        for i, fam in enumerate(families):
            pc_rows.append({"model": m, "family": fam, "precision": p[i], "recall": r[i],
                            "f1": f[i], "support": int(sup[i])})
    pd.DataFrame(pc_rows).to_csv(OUT / "per_class_metrics.csv", index=False, float_format="%.4f")
    (OUT / "classification_report_fl.txt").write_text(
        "Federated global model on global_test.csv (n=%d)\n\n" % len(y_te)
        + classification_report(y_te, preds[FL], labels=range(C.N_CLASSES),
                                target_names=families, digits=3, zero_division=0))

    # ---- which test samples does FL get wrong? (for the error analysis in the report) --
    errors = meta.assign(predicted=[families[p] for p in preds[FL]])
    errors = errors[errors.family != errors.predicted]
    errors.to_csv(OUT / "fl_misclassified.csv", index=False)

    # ---- paired comparisons: is FL really better / worse? -----------------------------
    comparisons = {"FL vs centralised": paired_difference(boot, FL, CEN)}
    for k, m in enumerate(LOCAL, 1):
        comparisons["FL vs local-only client %d" % k] = paired_difference(boot, FL, m)

    # ---- everything in one JSON ---------------------------------------------------------
    report = {
        "fl_model": _rel(args.fl_model), "n_global_test": int(len(y_te)),
        "n_high_confidence": int(hc.sum()), "families": families, "bootstrap_resamples": n_boot,
        "metrics": summary.set_index("model").to_dict(orient="index"),
        "paired_comparisons_macro_f1": comparisons,
        "confusion_matrices": {m: cms[m].tolist() for m in MODELS},
        "test_label_agreement_by_family": pd.crosstab(meta.family, meta.label_agreement)
        .to_dict(orient="index"),
        # error rate of the FL model per label-agreement group: are the "wrong" predictions
        # concentrated on samples whose Midterm label AVClass does not confirm?
        "fl_error_rate_by_label_agreement": {
            g: {"errors": int((errors.label_agreement == g).sum()),
                "samples": int((meta.label_agreement == g).sum()),
                "rate": float((errors.label_agreement == g).sum()
                              / (meta.label_agreement == g).sum())}
            for g in sorted(meta.label_agreement.unique())},
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- figures --------------------------------------------------------------------------
    fig_confusion(cms[FL], families, FIG / "fl_confusion_matrix.png",
                  "Federated global model: confusion matrix on the global test set (n = 54)")
    fig_confusion_grid(cms, families, FIG / "confusion_matrices_all_models.png")
    fig_comparison(summary, FIG / "model_comparison.png", n_boot)
    fig_per_class(per_class, families, FIG / "per_class_f1.png")
    if args.fl_rounds and Path(args.fl_rounds).exists():
        fig_convergence(pd.read_csv(args.fl_rounds), summary, FIG / "fl_convergence.png")
    else:
        print("no server_rounds.csv found -> convergence figure skipped")

    # ---- console summary ----------------------------------------------------------------
    print("\n%-22s %6s %6s %6s %6s   %-13s  %s" % ("model", "acc", "prec", "rec", "F1",
                                                 "F1 95% CI", "F1 high-conf."))
    for _, r in summary.iterrows():
        print("%-22s %6.3f %6.3f %6.3f %6.3f   [%.2f, %.2f]   %.3f" % (
            r.model, r.accuracy, r.precision_macro, r.recall_macro, r.f1_macro,
            r.f1_macro_ci_low, r.f1_macro_ci_high, r.hc_f1_macro))
    print("\npaired bootstrap, macro-F1 difference:")
    for name, c in comparisons.items():
        print("  %-28s %+.3f  95%% CI [%+.3f, %+.3f]  P(FL better)=%.2f  %s" % (
            name, c["mean"], c["ci95"][0], c["ci95"][1], c["p_a_better"],
            "significant" if c["significant"] else "not significant"))
    print("\nFL error rate by label agreement:")
    for g, e in report["fl_error_rate_by_label_agreement"].items():
        print("  %-11s %2d / %2d = %.2f" % (g, e["errors"], e["samples"], e["rate"]))
    print("high-confidence subset (agree/partial): n = %d" % hc.sum())
    print("saved to %s  (%.0f s)" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
