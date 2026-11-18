# HW4 – Member 5: Evaluation of the federated global model

`evaluate.py` tests the final FedAvg global model on `data/clients/global_test.csv` (54 samples,
never used for training) and compares it with local-only and centralised training.

**Status:** the numbers below come from Member 2's 3-client simulation
(`results/member2_client1/fl_simulation/global_model.keras`). They are a stand-in until
Member 4's final server run exists; re-run the script on that model (see below).

## What it computes
| Item | Details |
|---|---|
| Models | Federated (loaded), centralised and local-only clients 1–3 (re-trained here with `fl_common.build_model()`, same split and the same 100-epoch budget: 20 rounds × 5 epochs) |
| Metrics | accuracy, precision, recall, F1 (macro and weighted), per family, confusion matrices |
| Uncertainty | 95 % bootstrap CI (2,000 resamples of the 54 test samples); *paired* bootstrap for FL vs. each baseline |
| Label noise | the same metrics on the high-confidence subset (`label_agreement` = agree/partial, n = 31) and the FL error rate per agreement group |

The re-trained baselines reproduce Member 2's `baselines.csv` exactly, so the setup is identical.

## Run
```bash
# from the repo root, in the env of member2_client1/requirements-fl.txt (Python 3.11)
python member5_evaluation/evaluate.py                       # simulation model (default)
python member5_evaluation/evaluate.py --fl-model PATH/global_model.keras --fl-rounds PATH/server_rounds.csv
```
`--fl-model` also accepts an `.npz` with the 6 weight arrays in layer order. About 30 s on CPU.

## Outputs (`results/member5_evaluation/`)
| File | Content |
|---|---|
| `metrics_summary.csv` | one row per model: metrics, 95 % CIs, high-confidence accuracy/F1 |
| `per_class_metrics.csv` | precision/recall/F1/support per model and family |
| `metrics.json` | everything above + confusion matrices, paired comparisons, label-agreement analysis |
| `classification_report_fl.txt` | scikit-learn report of the FL model |
| `fl_misclassified.csv` | the test samples the FL model gets wrong (sample_id, true/predicted family, label agreement) |
| `figures/fl_confusion_matrix.png` | FL confusion matrix, counts and row-normalised |
| `figures/confusion_matrices_all_models.png` | all five models side by side |
| `figures/model_comparison.png` | 4 metrics × 5 models with CI whiskers |
| `figures/per_class_f1.png` | F1 per family: FL, centralised, mean of local-only |
| `figures/fl_convergence.png` | macro-F1 of the global model per FedAvg round |

## Results (simulation model, 19 Nov 2026)
| Model (global test, n = 54) | Accuracy | Precision | Recall | Macro-F1 | F1 95 % CI |
|---|---:|---:|---:|---:|---|
| **Federated (FedAvg)** | **0.704** | **0.726** | **0.697** | **0.704** | [0.56, 0.82] |
| Centralised | 0.704 | 0.712 | 0.693 | 0.695 | [0.55, 0.81] |
| Local-only client 1 | 0.593 | 0.585 | 0.585 | 0.585 | [0.45, 0.70] |
| Local-only client 2 | 0.611 | 0.665 | 0.617 | 0.616 | [0.48, 0.74] |
| Local-only client 3 | 0.556 | 0.575 | 0.544 | 0.542 | [0.40, 0.66] |

**Paired bootstrap (macro-F1, FL minus baseline):**
- vs. centralised: +0.008, 95 % CI [−0.09, +0.11] → no measurable difference: FL reaches the centralised result without sharing raw data.
- vs. local-only clients 1 / 2 / 3: +0.12 / +0.09 / +0.16; significant for clients 1 and 3, CI touches 0 for client 2.

**How FL performs:**
- FL converges in about 12–15 rounds and passes the local-only mean already after round 4 (`fl_convergence.png`).
- The gain over local-only training is largest for LokiBot (F1 0.62 vs. 0.37 local-only mean), the smallest family (~11 samples per client).
- Main confusions: formbook/lokibot/asyncrat predicted as agenttesla (8 of 16 errors).

**Label noise limits all models, not FL in particular:** the FL error rate is 0.19 on samples whose
label AVClass confirms (agree), but 0.41 on `disagree` samples. 8 of 9 LokiBot test samples are `disagree`.
On the high-confidence subset (n = 31) accuracy rises to 0.77 (FL) / 0.81 (centralised).
Macro-F1 on this subset is unstable because it contains only one LokiBot sample.

**Limitations:** one test set of 54 samples (CIs are ±0.13 wide); one training seed per model;
IID split only, so the result says nothing about non-IID clients.
