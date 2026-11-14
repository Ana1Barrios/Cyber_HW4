# HW4 - Dataset and client distribution (Member 1)

## 1. Source and cleaning
Inputs come from the Midterm (hybrid analysis = API-call sequences + network traffic), read from the Drive:
`api_sequences.jsonl` (270 samples), `network_dataset_labeled_270.csv` (270), `samples.csv` (metadata).

| Step | Result |
|---|---|
| Samples in both sources | 270 (5 families: agenttesla, amadey, asyncrat, formbook, lokibot) |
| Invalid pcap (status `empty`, 0 packets) | **S0001 removed** (all-zero network features) |
| Duplicated sha256 | 0 |
| NaN / inf values | 0 (the script still imputes with the median as a safety net) |
| Label mismatch API vs network | 0 (asserted) |
| **Final dataset** | **269 samples** (same 269 used in the Midterm hybrid) |

The Midterm `train/val/test` column is **not** reused: its test split has 0 LokiBot samples and only 32 rows,
so it cannot evaluate all 5 classes.

## 2. Features (1,040, identical for all clients)
FedAvg needs the same model shape on every client, so the feature space is fixed and stateless:
- **API (1,029):** 1-3-grams of the API sequence hashed into 1,024 buckets (`HashingVectorizer`, no vocabulary to share),
  `log1p` of the counts, L2-normalised; plus 5 statistics (`n_calls`, `n_collapsed`, `n_procs`, `n_unique_apis`, `api_entropy`).
- **Network (11):** packet/byte counts, duration, mean packet size, TCP/UDP/DNS/HTTP/TLS/ICMP/other packets.
- Heavy-tailed counts get `log1p`; numeric statistics are standardised with mean/std from the **training pool only**
  (`preprocessing_params.json`). The global test set never influences preprocessing.
- Differences vs. the Midterm: hashing replaces TF-IDF (TF-IDF needs a global vocabulary and IDF, which clients would have to share),
  and the network features enter directly (early fusion) instead of as network-model probabilities.

## 3. Split (IID, stratified, seed 42)
1. 20 % **global test set** (54 samples), held out; used only by Member 5 for the final evaluation.
2. The remaining 215 samples are split in 3 **IID** clients with `StratifiedKFold(3)`, so every client has all 5 families in similar proportion.

| Family | Client 1 | Client 2 | Client 3 | Global test | Total |
|---|---:|---:|---:|---:|---:|
| agenttesla | 16 | 15 | 15 | 12 | 58 |
| amadey | 15 | 15 | 14 | 11 | 55 |
| asyncrat | 14 | 14 | 15 | 11 | 54 |
| formbook | 16 | 16 | 15 | 11 | 58 |
| lokibot | 11 | 12 | 12 | 9 | 44 |
| **Total** | **72** | **72** | **71** | **54** | **269** |

Figure: `reports/figures/client_distribution.png`. Table: `docs/client_distribution.csv`.

## 4. Reference numbers (sanity check only, not the FL result)
Logistic regression (C=10, balanced) evaluated on the global test set (n=54):

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Centralised (3 clients pooled) | 0.722 | 0.716 |
| Local only, client 1 | 0.685 | 0.676 |
| Local only, client 2 | 0.574 | 0.567 |
| Local only, client 3 | 0.574 | 0.560 |

Pooled accuracy is in line with the Midterm hybrid (0.73). With ~70 samples per client, local models are weaker than the pooled one:
this is the gap FL should close. The test set is small (54), so differences of a few points are within noise; report this in the evaluation.

## 5. Notes and limitations
- Small data: ~14 samples per family per client.
- Label noise: only 53.5 % of the Midterm labels are confirmed by AVClass (`label_agreement` column is kept in each file as metadata, not as a feature), so a high-confidence subset (`agree` + `partial`) can be analysed separately.
- Scaler statistics are computed on the pooled training data for simplicity. In a real deployment they would be aggregated from per-client sums or fixed a priori.
- Client files contain no raw traces, pcaps or binaries, only numeric features and sample IDs.
