"""
train_all_models.py
===================
Unified training pipeline for three-class sentiment analysis.

Models included:
  1. Logistic Regression (baseline)
  2. TextCNN              (Kim 2014)
  3. BiLSTM               (bidirectional LSTM)
  4. DistilBERT           (distilbert-base-uncased)
  5. BERT-base            (bert-base-uncased)

Usage:
  python train_all_models.py --model all
  python train_all_models.py --model lr
  python train_all_models.py --model textcnn
  python train_all_models.py --model lr textcnn
  python train_all_models.py --model bilstm
  python train_all_models.py --model distilbert
  python train_all_models.py --model bert

Requires:
  Data.csv  (columns: movie_id, title, review, rating)
"""

import argparse
import json
import os

import numpy as np
import torch

from data_handler import (
    load_and_split_data, compute_weights, RESULTS_DIR, device
)
from visualization import print_and_save_comparison, plot_performance

RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument(
        '--model', type=str, nargs='+', default=['all'],
        choices=['all', 'lr', 'textcnn', 'bilstm', 'distilbert', 'bert'],
        help='Which model(s) to run'
    )
    parser.add_argument(
        '--scheme', type=str, default='default',
        choices=['default', 'wide_neutral', 'narrow_neutral'],
        help='Label threshold scheme'
    )

    parser.add_argument(
        '--force_resplit', action='store_true',
        help='Force a fresh train/val/test split even if CSVs already exist'
    )


    
    args = parser.parse_args()

    print(f"\nDevice      : {device}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Model(s)    : {', '.join(args.model)}")
    print(f"Label scheme: {args.scheme}")

    train_df, val_df, test_df = load_and_split_data(
        scheme=args.scheme, force_resplit=args.force_resplit
    )
    weight_dict, weight_tensor = compute_weights(train_df)

    all_results   = {}
    models_to_run = set(args.model)
    run_all       = 'all' in models_to_run

    if run_all or 'lr' in models_to_run:
        from model_lr import run_logistic_regression
        all_results['Logistic Regression'] = run_logistic_regression(
            train_df, val_df, test_df, weight_dict
        )

    if run_all or 'textcnn' in models_to_run:
        from model_textcnn import run_textcnn
        all_results['TextCNN'] = run_textcnn(
            train_df, val_df, test_df, weight_tensor
        )

    if run_all or 'bilstm' in models_to_run:
        from model_bilstm import run_bilstm
        all_results['BiLSTM'] = run_bilstm(
            train_df, val_df, test_df, weight_tensor
        )

    if run_all or 'distilbert' in models_to_run:
        from model_distilbert import run_distilbert
        all_results['DistilBERT'] = run_distilbert(
            train_df, val_df, test_df, weight_tensor
        )

    if run_all or 'bert' in models_to_run:
        from model_bert import run_bert
        all_results['BERT'] = run_bert(
            train_df, val_df, test_df, weight_tensor
        )

    if all_results:
        csv_path   = os.path.join(RESULTS_DIR, 'final_comparison.csv')
        chart_path = os.path.join(RESULTS_DIR, 'final_performance.png')
        print_and_save_comparison(all_results, csv_path=csv_path)
        plot_performance(all_results, chart_path)

        summary_path = os.path.join(RESULTS_DIR, 'all_results.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved → {summary_path}")

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
