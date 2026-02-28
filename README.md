# timesnet-bft-datasets
# TimesNet-BFT: Open Simulation Datasets

This repository contains the simulation datasets used in the paper **"TimesNet-BFT: Mitigating Network State Uncertainty in Byzantine Consensus via Deep Temporal Modeling"**. 

These latency traces are provided to reproduce the temporal modeling and BFT consensus evaluation experiments described in our study.

## 📂 Dataset Description

The repository provides 3 synthetic `.csv` datasets, formatted as `(Views, Nodes)` where $N=32$.

* **`latency_main_1k.csv` (1000 × 32):** The primary discrete-event simulation log representing 1000 consensus views. It features dynamic network latency (superposition of base delay, dual-frequency wave, and Poisson congestion) and deterministic macroscopic anomalies (25% node failure at 600ms boundary).
* **`latency_wan.csv` (200 × 32):** A Zero-Shot generalization benchmark simulating Wide Area Network (WAN) conditions, characterized by higher base propagation latencies (100-250 ms) and route flapping events.
* **`latency_pretrain_100k.7z` (100,000 × 32):** An extended synthetic pre-training corpus (compressed). Designed for offline transfer learning, it provides the massive data density required to initialize deep temporal models before fine-tuning on the primary consensus logs.

## 🔗 External Real-World Dataset (VeReMi)

In addition to the synthetic datasets, our study conducts zero-shot evaluations using the public **VeReMi (Vehicular Reference Misbehavior)** dataset. The raw message logs can be accessed directly from the official VeReMi repository:
* **Official Link:** [https://veremi-dataset.github.io/](https://veremi-dataset.github.io/)

## ⚙️ Usage Example (Python / Pandas)

```python
import pandas as pd

df_train = pd.read_csv('latency_train_full.csv')
print(f"Loaded training data with shape: {df_train.shape}")
