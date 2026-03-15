# Nuanced Sentiment Classification

Three-class sentiment analysis on IMDb movie reviews using Logistic Regression, TextCNN, BiLSTM, DistilBERT, and BERT-base.

## Setup

Install dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

Place `Data.csv` in the project root. Expected columns: `movie_id`, `title`, `review`, `rating`.

---

## Project Structure

```
├── train_all_models.py      # Entry point
├── data_handler.py          # Data loading, splitting, vocab, dataloaders
├── training_utils.py        # Shared neural training loop
├── visualization.py         # Plots and comparison tables
├── model_lr.py              # Logistic Regression
├── model_textcnn.py         # TextCNN
├── model_bilstm.py          # BiLSTM
├── model_distilbert.py      # DistilBERT
├── model_bert.py            # BERT-base
├── Data.csv                 # Raw dataset (not included)
└── results/                 # Output directory for models, plots, and JSON results
```

---

## Running the Pipeline

```bash
python train_all_models.py --model <model> --scheme <scheme>
```

### `--model` options

| Value | Description |
|---|---|
| `all` | Run all models |
| `lr` | Logistic Regression (TF-IDF baseline) |
| `textcnn` | TextCNN (Kim 2014) |
| `bilstm` | Bidirectional LSTM |
| `distilbert` | DistilBERT (distilbert-base-uncased) |
| `bert` | BERT-base (bert-base-uncased) |

You can pass multiple models at once:

```bash
python train_all_models.py --model lr textcnn bilstm
```

### `--scheme` options

| Value | Thresholds |
|---|---|
| `default` *(default)* | 1-4 bad, 5-6 neutral, 7-10 good |
| `wide_neutral` | 1-3 bad, 4-7 neutral, 8-10 good |
| `narrow_neutral` | 1-5 bad, 6 neutral, 7-10 good |

### `--force_resplit`

By default the pipeline reuses existing `train_expanded.csv`, `val_expanded.csv`, and `test_expanded.csv` if they are already on disk. Pass this flag to regenerate them from scratch:

```bash
python train_all_models.py --model all --force_resplit
```

---

## Examples

```bash
# Run everything with wide neutral scheme
python train_all_models.py --model all --scheme wide_neutral

# Baseline only
python train_all_models.py --model lr

# Label threshold ablation (run each scheme with LR)
python train_all_models.py --model lr --scheme narrow_neutral --force_resplit
python train_all_models.py --model lr --scheme default --force_resplit
python train_all_models.py --model lr --scheme wide_neutral --force_resplit

# TextCNN + BiLSTM with wide neutral
python train_all_models.py --model textcnn bilstm --scheme wide_neutral

# Force a fresh data split then run BERT
python train_all_models.py --model bert --force_resplit
```

---

## Outputs

All results are saved to the `results/` directory:

- `all_results.json` — full metrics for every model and condition
- `final_comparison.csv` — summary table
- `final_performance.png` — grouped bar chart
- `{model}_{condition}_curves.png` — training/validation curves per model
- `{model}_{condition}_confusion.png` — confusion matrix per model
- `{model}_{condition}_best.pt` — best model checkpoint per run