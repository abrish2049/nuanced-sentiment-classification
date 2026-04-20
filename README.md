# From Binary to Nuanced: Three-Class Sentiment Analysis in Movie Reviews

This repository contains the code for the paper *"From Binary to Nuanced: A Deep Learning Study of Three-Class Sentiment Analysis in Movie Reviews"* (Veliveli, Ventura, Maloon, Tedla — University of Virginia, 2026).

We reframe IMDb sentiment analysis as a three-class problem (bad / neutral / good) using rating annotations as weak supervision. Five model families are evaluated with ablations on label thresholds, class weighting, focal loss, fine-tuning strategy, sequence length, and GloVe embeddings across three random seeds.

**Best result:** BERT-base with class weighting or last-2-layer fine-tuning — macro-F1 **0.784 ± 0.002**, accuracy **81.7 ± 0.5%**

---

## Key Results Summary

| Model | Variant | Acc | Macro-F1 | F1 Neutral |
|---|---|---|---|---|
| LR | No weighting | .793 | .741 | .536 |
| LR | Class weights | .782 | .750 | .582 |
| TextCNN | No weighting | .749±.001 | .708±.002 | .508±.005 |
| TextCNN | Class weights | .742±.009 | .708±.004 | .520±.005 |
| BiLSTM | No weighting | .773±.002 | .723±.002 | .513±.004 |
| BiLSTM | Class weights | .755±.003 | .723±.003 | .545±.006 |
| DistilBERT | No weighting | .818±.003 | .779±.008 | .605±.021 |
| DistilBERT | Class weights | .815±.003 | .779±.003 | .612±.005 |
| BERT | Class weights | .817±.005 | **.784±.002** | .625±.008 |
| BERT | Focal (γ=2) | .811±.003 | .783±.003 | **.630±.008** |
| BERT | Last-2-layers | .817±.002 | **.784±.001** | .624±.002 |
| BERT | Head-only | .705±.002 | .672±.001 | .496±.001 |

Binary reference (LR, bad/good only): Acc = 0.924, Macro-F1 = 0.923

---

## Dataset

108,133 IMDb reviews with ratings 1–10. Label scheme: **wide-neutral** — Bad (1–3), Neutral (4–7), Good (8–10).

| Class | Count | % |
|---|---|---|
| Bad | 39,673 | 36.7 |
| Neutral | 21,199 | 19.6 |
| Good | 47,261 | 43.7 |

Split: train 75,693 / val 16,220 / test 16,220 (stratified).

`Data.csv` is included in the repository. Expected columns: `movie_id`, `title`, `review`, `rating`.

---

## Setup

```bash
pip install -r requirements.txt
```

**Optional — GloVe embeddings** (only needed for the TextCNN + GloVe ablation, ~822 MB). Run on the HPC login node before submitting the job, since compute nodes block outbound internet:

```bash
python download_glove.py
```

---

## Project Structure

```
├── train_all_models.py      # Entry point — unified training pipeline
├── data_handler.py          # Data loading, splitting, vocab, class weights
├── training_utils.py        # Shared neural training loop + history CSV export
├── losses.py                # Focal loss (γ=2)
├── visualization.py         # Training curves, confusion matrices, TF-IDF charts
├── model_lr.py              # Logistic Regression + TF-IDF feature analysis
├── model_textcnn.py         # TextCNN (Kim 2014), optional GloVe embeddings
├── model_bilstm.py          # Bidirectional LSTM
├── model_distilbert.py      # DistilBERT fine-tuning
├── model_bert.py            # BERT-base + freeze/focal ablations + attention viz
├── error_analysis.py        # Qualitative error analysis on misclassified reviews
├── all_models_score_viz.py  # Accuracy vs F1 scatter + BERT ablation chart
├── download_glove.py        # Standalone GloVe downloader (run on login node)
├── run.slurm                # HPC batch job — runs all experiments
├── test_pipeline.py         # Smoke-test suite (pytest)
├── Data.csv                 # Raw dataset (108K IMDb reviews with ratings 1–10)
└── results/                 # All outputs — organised by results/len{max_len}/
```

---

## Reproducing Paper Tables

### Table II — Main results (all models, wide-neutral, max_len=512, 3 seeds)

```bash
python train_all_models.py --model all --scheme wide_neutral --max_len 512 --seeds 42 43 44
```

Output: `results/len512/final_comparison_len512.csv`

### Table III — Label threshold ablation (LR across 3 schemes)

```bash
python train_all_models.py --model lr --scheme narrow_neutral --force_resplit --results_dir results/threshold_ablation/narrow_neutral
python train_all_models.py --model lr --scheme default         --force_resplit --results_dir results/threshold_ablation/default
python train_all_models.py --model lr --scheme wide_neutral    --force_resplit --results_dir results/threshold_ablation/wide_neutral
```

> `--force_resplit` is required because each scheme remaps class labels and the splits must be regenerated.

### Table IV — Sequence length ablation (BiLSTM, DistilBERT, BERT)

```bash
python train_all_models.py --model bilstm distilbert bert --scheme wide_neutral --max_len 128 --seeds 42 43 44
python train_all_models.py --model bilstm distilbert bert --scheme wide_neutral --max_len 256 --seeds 42 43 44
python train_all_models.py --model bilstm distilbert bert --scheme wide_neutral --max_len 512 --seeds 42 43 44
```

### Table V — BERT fine-tuning strategy ablation (head-only, last-2-layers, full)

Included in the main Table II run above — BERT is evaluated under all five variants in one pass.

### Table VI — TextCNN GloVe ablation

```bash
python train_all_models.py --model textcnn --scheme wide_neutral --glove_path glove.6B.300d.txt
```

Runs both random-embedding and GloVe-300d-frozen variants in a single pass.

---

## Post-Training Analysis

### Qualitative error analysis (Section VII)

Error CSVs are produced automatically during training. To generate the 30-review inspection report:

```bash
# BERT (class-weighted, len=512, seed 42) — used in the paper
python error_analysis.py --error_csv results/len512/bert_len512_with_weighting_errors.csv --output results/len512/bert_error_report.txt

# LR
python error_analysis.py --error_csv results/len512/lr_errors.csv --output results/len512/lr_error_report.txt
```

### Visualise results (Figure 2)

```bash
python all_models_score_viz.py --max_len 512
```

### Run tests

```bash
pytest test_pipeline.py -v
```

---

## Argument Reference

| Argument | Values | Default | Description |
|---|---|---|---|
| `--model` | `all lr textcnn bilstm distilbert bert` | `all` | Model(s) to run. Multiple allowed. |
| `--scheme` | `default wide_neutral narrow_neutral` | `default` | Label threshold scheme. |
| `--max_len` | `128 256 512` | `512` | Max token sequence length. |
| `--seeds` | one or more ints | `42` | Random seeds. Pass 3 for mean ± std. |
| `--glove_path` | path to `.txt` | `None` | GloVe vectors file. Enables frozen-GloVe TextCNN variants. |
| `--force_resplit` | flag | off | Regenerate train/val/test CSVs from scratch. |
| `--results_dir` | path | `results/len{max_len}` | Override output directory. |

---

## Outputs

All results are saved to `results/len{max_len}/` (or `results/len{max_len}/seed{seed}/` for multi-seed runs):

| File | Description |
|---|---|
| `all_results_len{max_len}.json` | Full metrics for every model and variant |
| `final_comparison_len{max_len}.csv` | Summary table (accuracy, macro-F1, neutral F1) |
| `multiseed_summary_len{max_len}.json` | Mean ± std across seeds |
| `final_performance_len{max_len}.png` | Grouped bar chart comparing all models |
| `rating_distribution.png` | Histogram of raw 1–10 IMDb ratings (Fig. 1) |
| `{tag}_curves.png` | Training and validation loss/F1 curves (Fig. 5, 6) |
| `{tag}_confusion.png` | 3×3 confusion matrix (Fig. 3, 4, 8) |
| `{tag}_history.csv` | Per-epoch metrics + final test row |
| `{tag}_errors.csv` | All misclassified test examples |
| `{tag}_best.pt` | Best model checkpoint (by val macro-F1) |
| `lr_tfidf_features.png` | Top TF-IDF coefficient words per class (Fig. 9) |
| `bert_attention_neutral_correct_*.png` | CLS attention — correctly classified neutral reviews |
| `bert_attention_neutral_wrong_*.png` | CLS attention — misclassified neutral reviews |

---

## Citation

> S. Veliveli, A. J. Ventura, P. Maloon, A. Tedla, "From Binary to Nuanced: A Deep Learning Study of Three-Class Sentiment Analysis in Movie Reviews," University of Virginia, 2026.

---

## Note on Pre-computed Results

Due to compute constraints, experiments were run across multiple sessions. Model checkpoints (`.pt` files) have been removed to reduce repository size. All metrics, plots, and CSV outputs have been consolidated and are available as `results.zip` in the root of the repository.

> **Note on folder structure:** Because experiments were submitted as separate jobs, the layout inside `results.zip` differs from the `results/len{max_len}/seed{seed}/` hierarchy that `train_all_models.py` produces when run from scratch. File contents and naming conventions are otherwise identical.

### Estimated Compute Time

Running all models across all three seeds on a single NVIDIA A100 GPU takes approximately **28 hours** of total GPU compute, split across multiple jobs:

| Experiment | Models | Wall Time |
|---|---|---|
| All models, len=512, 3 seeds | LR, TextCNN, BiLSTM, DistilBERT, BERT | ~13h 20m |
| BiLSTM + DistilBERT + BERT, len=256, 3 seeds | — | ~6h 42m |
| BiLSTM + DistilBERT + BERT, len=128, 3 seeds | — | ~4h 09m |
| TextCNN GloVe ablation, len=512 | TextCNN (4 variants) | ~4h |

If submitting as a single job, request at least `--time=28:00:00`. The `run.slurm` script is currently set to `16:00:00` and is intended to be split across separate submissions.
