# Sentiment Analysis Pipeline

## Setup

Install dependencies:
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

Place `Data.csv` in the project root. Expected columns: `movie_id`, `title`, `review`, `rating`.

---

## Running the pipeline

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
| `default` *(default)* | 1–4 → bad, 5–6 → neutral, 7–10 → good |
| `wide_neutral` | 1–3 → bad, 4–7 → neutral, 8–10 → good |

### `--force_resplit`

By default the pipeline reuses existing `train_expanded.csv`, `val_expanded.csv`, and `test_expanded.csv` if they are already on disk. Pass this flag to regenerate them from scratch:
```bash
python train_all_models.py --model all --force_resplit
```

---

## Examples

```bash
# Run everything with default label scheme
python train_all_models.py --model all

# Run everything with wide neutral band
python train_all_models.py --model all --scheme wide_neutral

# Baseline only
python train_all_models.py --model lr

# TextCNN + BiLSTM with wide neutral
python train_all_models.py --model textcnn bilstm --scheme wide_neutral

# Force a fresh data split then run BERT
python train_all_models.py --model bert --force_resplit
```

