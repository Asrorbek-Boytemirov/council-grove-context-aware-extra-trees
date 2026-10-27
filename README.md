# Context-Aware Extra Trees for Unseen-Well Lithofacies Classification

This repository contains the executable code, Council Grove dataset, dependencies, and generated results for a leakage-safe context-aware Extra Trees benchmark.

## Experiment

The code evaluates contextual features under row-wise stratified cross-validation and leave-one-well-out validation. The outputs include class-wise and aggregate metrics, per-well performance, confidence intervals, statistical tests, confusion matrices, computational cost, figures, and out-of-sample predictions.

## Repository files

- `council_grove_benchmark.py` — complete experiment program.
- `requirements.txt` — Python dependencies.
- `facies_vectors.csv` — Council Grove dataset copy used by the benchmark.
- `Council_Grove_Code_Data_Results.zip` — code, dataset, complete result tables, figures, predictions, and software-environment metadata.

## Reproduction

Use Python 3.11 or later:

```bash
python -m pip install -r requirements.txt
python council_grove_benchmark.py
```

The script verifies the dataset SHA-256 digest, trains the declared models under the documented protocols, and writes all generated outputs to `council_grove_results/`.

## Headline result

The context-aware Extra Trees model achieved approximately 0.799 accuracy under row-wise stratified cross-validation and 0.551 under leave-one-well-out validation. The difference quantifies the optimism that can arise when row-wise validation is used for an unseen-well deployment target.
