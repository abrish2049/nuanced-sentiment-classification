# Nuanced Sentiment Classification

Three-class sentiment analysis (bad, neutral, good) on 108K IMDb movie reviews. The project reframes the standard binary IMDb task by retaining middle-rated reviews (ratings 4–7) as a neutral class, using rating annotations as weak supervision. Five model families are evaluated — Logistic Regression, TextCNN, BiLSTM, DistilBERT, and BERT-base — with ablations on class weighting, label thresholds, sequence length, fine-tuning strategy, and focal loss. The neutral class is the primary bottleneck across all models (F1 0.51–0.62); adding it costs roughly 10 accuracy points compared to binary classification.

## Setup

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

Place `Data.csv` in the project root. Expected columns: `movie_id`, `title`, `review`, `rating`.

## Project Structure

```
├── train_all_models.py      # Entry point — unified training pipeline
├── data_handler.py          # Data loading, splitting, vocab, weights, dataloaders
├── training_utils.py        # Shared neural training loop + history CSV export
├── visualization.py         # Plots, confusion matrices, comparison tables
├── losses.py                # Focal loss implementation
├── model_lr.py              # Logistic Regression + feature/error analysis
├── model_textcnn.py         # TextCNN (Kim 2014)
├── model_bilstm.py          # Bidirectional LSTM
├── model_distilbert.py      # DistilBERT fine-tuning
├── model_bert.py            # BERT-base + freeze/focal ablations + attention viz
├── all_models_score_viz.py  # Acc vs F1 scatter plot from saved results
├── Data.csv                 # Raw dataset (not included in repo)
└── results/                 # All outputs: checkpoints, plots, CSVs, JSONs
```

## Usage

```bash
python train_all_models.py [OPTIONS]
```

| Argument | Values | Default | Description |
|---|---|---|---|
| `--model` | `all lr textcnn bilstm distilbert bert` | `all` | Model(s) to run. Multiple allowed. |
| `--scheme` | `default wide_neutral narrow_neutral` | `default` | Label threshold scheme. |
| `--max_len` | `128 256 512` | `512` | Max token sequence length (BERT, DistilBERT, BiLSTM). |
| `--force_resplit` | flag | off | Regenerate train/val/test CSVs from scratch. |

```bash
# Run all models with wide neutral scheme
python train_all_models.py --model all --scheme wide_neutral

# Sequence length ablation for BERT
python train_all_models.py --model bert --max_len 128
python train_all_models.py --model bert --max_len 256
python train_all_models.py --model bert --max_len 512

# Label threshold ablation with LR
python train_all_models.py --model lr --scheme narrow_neutral --force_resplit
python train_all_models.py --model lr --scheme default --force_resplit
python train_all_models.py --model lr --scheme wide_neutral --force_resplit

# Visualize results
python all_models_score_viz.py [--max_len 512]
```

## Outputs

All results go to `results/`:

- `all_results_len{max_len}.json` — full metrics for every model
- `final_comparison.csv` / `final_performance.png` — summary table and chart
- `{tag}_curves.png` / `{tag}_confusion.png` — per-run training curves and confusion matrices
- `{tag}_history.csv` — per-epoch train/val loss, accuracy, F1 + final test metrics
- `{tag}_errors.csv` — misclassified examples (BERT, LR)
- `bert_attention_neutral_*.png` — CLS attention visualizations for neutral reviews
- `{tag}_best.pt` — best model checkpoint per run
