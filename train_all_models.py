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
  python train_all_models.py --model all --scheme wide_neutral
  python train_all_models.py --model bert --max_len 128
  python train_all_models.py --model all --seeds 42 43 44      # mean±std
  python train_all_models.py --model textcnn --glove_path glove.6B.300d.txt

Requires:
  Data.csv  (columns: movie_id, title, review, rating)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from data_handler import (
    load_and_split_data, compute_weights, RESULTS_DIR, device
)
from visualization import (
    print_and_save_comparison, plot_performance, plot_rating_distribution
)

RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()


def _set_results_dir(results_dir):
    """Patch RESULTS_DIR in every pipeline module so all outputs from this
    run land in the correct directory. Safe to call multiple times (e.g. per
    seed) because it only touches modules that are already imported."""
    for mod_name in ['data_handler', 'visualization', 'training_utils',
                     'model_lr', 'model_bert', 'model_distilbert',
                     'model_bilstm', 'model_textcnn']:
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], 'RESULTS_DIR', results_dir)


def _aggregate_seeds(seed_results_list):
    """Compute mean ± std across per-seed result dicts.

    Parameters
    ----------
    seed_results_list : list of dict
        Each element is an all_results dict from one seed run:
        { model_name: { variant: { metric: scalar } } }

    Returns
    -------
    dict — same structure but scalars replaced with {'mean': .., 'std': ..}
    """
    if not seed_results_list:
        return {}

    all_models = set()
    for r in seed_results_list:
        all_models.update(r.keys())

    aggregated = {}
    for model_name in sorted(all_models):
        aggregated[model_name] = {}
        all_variants = set()
        for r in seed_results_list:
            if model_name in r:
                all_variants.update(r[model_name].keys())

        for variant in sorted(all_variants):
            if variant == 'binary_reference':
                continue
            accs, f1s, bad_f1s, neu_f1s, good_f1s = [], [], [], [], []
            for r in seed_results_list:
                if model_name in r and variant in r[model_name]:
                    v    = r[model_name][variant]
                    pcf1 = v.get('per_class_f1', {})
                    accs.append(v.get('test_accuracy', float('nan')))
                    f1s.append(v.get('test_macro_f1',  float('nan')))
                    bad_f1s.append(pcf1.get('bad',     float('nan')))
                    neu_f1s.append(pcf1.get('neutral', float('nan')))
                    good_f1s.append(pcf1.get('good',   float('nan')))

            def _ms(vals):
                return {'mean': float(np.nanmean(vals)),
                        'std':  float(np.nanstd(vals))}

            aggregated[model_name][variant] = {
                'test_accuracy': _ms(accs),
                'test_macro_f1': _ms(f1s),
                'per_class_f1': {
                    'bad':     _ms(bad_f1s),
                    'neutral': _ms(neu_f1s),
                    'good':    _ms(good_f1s),
                }
            }
    return aggregated


def _print_multiseed_summary(aggregated):
    """Print a formatted mean ± std table to stdout."""
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY  (mean ± std)")
    print("=" * 80)
    print(f"{'Model / Variant':<40} {'Accuracy':>14} {'Macro-F1':>14} {'Neutral-F1':>12}")
    print("-" * 80)
    for model_name, variants in aggregated.items():
        for variant, metrics in variants.items():
            label = f"{model_name} ({variant.replace('_', ' ')})"
            acc = metrics['test_accuracy']
            f1  = metrics['test_macro_f1']
            neu = metrics['per_class_f1']['neutral']
            print(f"{label:<40} "
                  f"{acc['mean']:.3f}±{acc['std']:.3f}   "
                  f"{f1['mean']:.3f}±{f1['std']:.3f}   "
                  f"{neu['mean']:.3f}±{neu['std']:.3f}")


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
    parser.add_argument(
        '--max_len', type=int, default=512,
        choices=[128, 256, 512],
        help='Max token sequence length for BERT/BiLSTM'
    )
    parser.add_argument(
        '--results_dir', type=str, default=None,
        help='Output directory for results. Defaults to results/len{max_len}'
    )
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[42],
        help='Random seed(s). Pass multiple (e.g. --seeds 42 43 44) for mean±std reporting.'
    )
    parser.add_argument(
        '--glove_path', type=str, default=None,
        help='Path to GloVe vectors file (e.g. glove.6B.300d.txt). '
             'Adds frozen-GloVe variants to TextCNN runs.'
    )

    args = parser.parse_args()

    base_results_dir = args.results_dir or os.path.join('results', f'len{args.max_len}')
    os.makedirs(base_results_dir, exist_ok=True)

    print(f"\nDevice      : {device}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Model(s)    : {', '.join(args.model)}")
    print(f"Label scheme: {args.scheme}")
    print(f"Max length  : {args.max_len}")
    print(f"Seeds       : {args.seeds}")
    print(f"Results dir : {base_results_dir}")
    if args.glove_path:
        print(f"GloVe path  : {args.glove_path}")

    train_df, val_df, test_df = load_and_split_data(
        scheme=args.scheme, force_resplit=args.force_resplit
    )
    weight_dict, weight_tensor = compute_weights(train_df)

    # Rating distribution histogram — written once to the base results dir
    plot_rating_distribution(
        train_df, val_df, test_df,
        os.path.join(base_results_dir, 'rating_distribution.png')
    )

    multi_seed       = len(args.seeds) > 1
    all_seed_results = []

    for seed in args.seeds:
        set_seed(seed)

        results_dir = (
            os.path.join(base_results_dir, f'seed{seed}')
            if multi_seed else base_results_dir
        )
        os.makedirs(results_dir, exist_ok=True)
        _set_results_dir(results_dir)

        if multi_seed:
            print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

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
                train_df, val_df, test_df, weight_tensor,
                glove_path=args.glove_path
            )

        if run_all or 'bilstm' in models_to_run:
            from model_bilstm import run_bilstm
            all_results['BiLSTM'] = run_bilstm(
                train_df, val_df, test_df, weight_tensor,
                max_len=args.max_len
            )

        if run_all or 'distilbert' in models_to_run:
            from model_distilbert import run_distilbert
            all_results['DistilBERT'] = run_distilbert(
                train_df, val_df, test_df, weight_tensor,
                max_len=args.max_len
            )

        if run_all or 'bert' in models_to_run:
            from model_bert import run_bert
            all_results['BERT'] = run_bert(
                train_df, val_df, test_df, weight_tensor,
                max_len=args.max_len
            )

        if all_results:
            csv_path   = os.path.join(results_dir, f'final_comparison_len{args.max_len}.csv')
            chart_path = os.path.join(results_dir, f'final_performance_len{args.max_len}.png')
            print_and_save_comparison(all_results, csv_path=csv_path)
            plot_performance(all_results, chart_path)

            summary_path = os.path.join(results_dir, f'all_results_len{args.max_len}.json')
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nFull results saved -> {summary_path}")

        all_seed_results.append(all_results)

    # Multi-seed aggregation
    if multi_seed and all_seed_results:
        aggregated = _aggregate_seeds(all_seed_results)
        _print_multiseed_summary(aggregated)
        agg_path = os.path.join(base_results_dir,
                                f'multiseed_summary_len{args.max_len}.json')
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nMulti-seed summary saved -> {agg_path}")

    print("\nPipeline complete.")


if __name__ == '__main__':
    main()
