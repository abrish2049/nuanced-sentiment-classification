# Nuanced Sentiment Classification

Three-class sentiment analysis (bad / neutral / good) on 108K IMDb movie reviews. The project reframes the standard binary IMDb task by retaining middle-rated reviews (ratings 4–7) as a neutral class using rating annotations as weak supervision. Five model families are evaluated — Logistic Regression, TextCNN, BiLSTM, DistilBERT, and BERT-base — with ablations on class weighting, label thresholds, sequence length, fine-tuning strategy, and focal loss.
---

## Setup

```bash
pip install -r requirements.txt
```

`Data.csv` is included in the repository. Expected columns: `movie_id`, `title`, `review`, `rating`.

**Optional — GloVe embeddings** (only needed for the TextCNN + GloVe ablation, ~822 MB). Run on the HPC login node before submitting the job, since compute nodes block outbound internet:

```bash
python download_glove.py
```

---

## Project Structure

```
├── train_all_models.py      # Entry point — unified training pipeline
├── data_handler.py          # Data loading, splitting, vocab, weights, GloVe download
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

### Table II — Main results (all models, wide-neutral scheme)

```bash
python train_all_models.py --model all --scheme wide_neutral --max_len 512
```

Output: `results/len512/final_comparison_len512.csv`

### Table III — Label threshold ablation (LR across 3 schemes)

```bash
python train_all_models.py --model lr --scheme narrow_neutral --force_resplit
python train_all_models.py --model lr --scheme default         --force_resplit
python train_all_models.py --model lr --scheme wide_neutral    --force_resplit
```

Output: `logistic_regression_results.json` inside each scheme's results folder.

> `--force_resplit` is required here because each scheme remaps the class labels, so the data splits must be regenerated from scratch for each run.

---

## All Experiments

### Sequence length ablation (BERT + BiLSTM)

```bash
python train_all_models.py --model bert bilstm --scheme wide_neutral --max_len 128
python train_all_models.py --model bert bilstm --scheme wide_neutral --max_len 256
python train_all_models.py --model bert bilstm --scheme wide_neutral --max_len 512
```

Results land in `results/len128/`, `results/len256/`, `results/len512/`.

### Multi-seed runs (mean ± std)

```bash
python train_all_models.py --model all --scheme wide_neutral --max_len 512 --seeds 42 43 44
```

Per-seed results go to `results/len512/seed42/` etc. Aggregated mean ± std table saved to `results/len512/multiseed_summary_len512.json`.

### TextCNN + GloVe-300d ablation

```bash
python train_all_models.py --model textcnn --scheme wide_neutral --glove_path glove.6B.300d.txt
```

Runs both random-embedding and GloVe-frozen variants in one pass.

### Qualitative error analysis

After training, inspect 30 misclassified reviews (10 per true class). The error CSVs are produced automatically during training.

```bash
# LR errors
python error_analysis.py --error_csv results/len512/lr_errors.csv --output results/len512/lr_error_report.txt

# BERT errors (512 / 256 / 128)
python error_analysis.py --error_csv results/len512/bert_len512_with_weighting_errors.csv --output results/len512/bert_error_report.txt
python error_analysis.py --error_csv results/len256/bert_len256_with_weighting_errors.csv --output results/len256/bert_error_report.txt
python error_analysis.py --error_csv results/len128/bert_len128_with_weighting_errors.csv --output results/len128/bert_error_report.txt
```

### Visualise results

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
| `--max_len` | `128 256 512` | `512` | Max token sequence length (BERT, DistilBERT, BiLSTM). |
| `--seeds` | one or more ints | `42` | Random seeds. Pass 3 for mean±std reporting. |
| `--glove_path` | path to `.txt` | `None` | GloVe vectors file. Adds frozen-GloVe variants to TextCNN. |
| `--force_resplit` | flag | off | Regenerate train/val/test CSVs from scratch. |
| `--results_dir` | path | `results/len{max_len}` | Override output directory. |

---

## Outputs

All results are saved to `results/len{max_len}/` (or `results/len{max_len}/seed{seed}/` for multi-seed runs):

| File | Description |
|---|---|
| `all_results_len{max_len}.json` | Full metrics for every model and variant |
| `final_comparison_len{max_len}.csv` | Summary table (accuracy, macro-F1, neutral F1) |
| `final_performance_len{max_len}.png` | Grouped bar chart comparing all models |
| `rating_distribution.png` | Histogram of raw 1–10 IMDb rating counts |
| `{tag}_curves.png` | Training and validation loss/F1 curves |
| `{tag}_confusion.png` | 3×3 confusion matrix |
| `{tag}_history.csv` | Per-epoch metrics + final test row |
| `{tag}_errors.csv` | All misclassified test examples (LR and BERT) |
| `{tag}_best.pt` | Best model checkpoint (by val macro-F1) |
| `lr_tfidf_features.png` | Top TF-IDF coefficient words per class |
| `bert_attention_neutral_correct_*.png` | CLS attention — correctly classified neutral reviews |
| `bert_attention_neutral_wrong_*.png` | CLS attention — misclassified neutral reviews |
| `multiseed_summary_len{max_len}.json` | Mean ± std across seeds (multi-seed runs only) |

---

