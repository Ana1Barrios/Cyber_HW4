# Cyber_HW4 - Malware classification with Federated Learning

Homework 4: replace the Midterm's centralised model (hybrid API + network) with Federated Learning (3 clients).

## Member 1 deliverables (this folder)
| Path | Content |
|---|---|
| `data/clients/client1.csv`, `client2.csv`, `client3.csv` | cleaned, preprocessed IID client datasets (columns: `sample_id, family, label, label_agreement`, then 1,040 features) |
| `data/clients/global_test.csv` | held-out stratified test set for the final evaluation (Member 5) |
| `data/clients/feature_columns.json` | feature list and `label_map` (family -> int) |
| `data/clients/preprocessing_params.json` | seed, scaler mean/std, cleaning log |
| `preprocessing/preprocess.py` | preprocessing + split code |
| `preprocessing/verify_and_plot.py` | integrity checks, figure and reference baseline |
| `docs/data_distribution.md` | how the data is distributed |

## Use in the client code
```python
import json, pandas as pd
feats = json.load(open("data/clients/feature_columns.json"))["feature_columns"]
d = pd.read_csv("data/clients/client1.csv")
X, y = d[feats].to_numpy("float32"), d["label"].to_numpy()   # 5 classes
```
All clients use the same 1,040 features and the same label mapping.

## Reproduce
Inputs are on the team Drive (not in this repo): `api_sequences.jsonl`, `network_dataset_labeled_270.csv`, `samples.csv`.
```bash
pip install -r requirements.txt
python preprocessing/preprocess.py --api-sequences PATH/api_sequences.jsonl \
       --network-csv PATH/network_dataset_labeled_270.csv --samples-csv PATH/samples.csv
python preprocessing/verify_and_plot.py
```
Never commit malware binaries, pcaps or raw traces.
