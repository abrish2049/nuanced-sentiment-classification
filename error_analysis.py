"""
error_analysis.py
=================
Sample and display misclassified reviews for qualitative error analysis.

Reads any *_errors.csv produced by model_lr.py or model_bert.py, samples
10 misclassified reviews per true class (30 total), and prints them in a
readable format for manual inspection.

Error categories to look for (fill in the brackets manually):
  [ ] mixed_sentiment      — "the acting was great but the plot was terrible"
  [ ] sarcasm              — positive words used ironically
  [ ] genuinely_ambiguous  — text that could plausibly belong to either class
  [ ] rating_inconsistency — reviewer writes positive text but gives 5/10

Usage:
  python error_analysis.py --error_csv results/len512/lr_errors.csv
  python error_analysis.py --error_csv results/len512/bert_len512_with_weighting_errors.csv
  python error_analysis.py --error_csv results/len512/lr_errors.csv --output report.txt
"""

import argparse
import textwrap

import pandas as pd


def sample_errors(error_df, n_per_class=10, seed=42):
    """Return up to n_per_class misclassified rows per true class."""
    classes = ['bad', 'neutral', 'good']
    parts   = []
    for cls in classes:
        subset = error_df[error_df['true'] == cls]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(n_per_class, len(subset)),
                                   random_state=seed))
    return pd.concat(parts, ignore_index=True)


def format_report(sample_df):
    lines = []
    lines.append("=" * 80)
    lines.append("QUALITATIVE ERROR ANALYSIS")
    lines.append(f"Showing {len(sample_df)} misclassified reviews")
    lines.append("=" * 80)

    for cls in ['bad', 'neutral', 'good']:
        subset = sample_df[sample_df['true'] == cls]
        if subset.empty:
            continue
        lines.append(f"\n{'─' * 60}")
        lines.append(f"TRUE CLASS: {cls.upper()}  ({len(subset)} samples)")
        lines.append(f"{'─' * 60}")

        for i, (_, row) in enumerate(subset.iterrows(), 1):
            rating  = row.get('rating', 'N/A')
            pred    = row['predicted']
            review  = str(row['review'])
            snippet = review[:400] + ('…' if len(review) > 400 else '')
            wrapped = textwrap.fill(snippet, width=78,
                                    initial_indent='  ',
                                    subsequent_indent='  ')
            lines.append(f"\n[{i}]  Rating: {rating}  |  True: {cls}  →  Predicted: {pred}")
            lines.append(wrapped)
            lines.append("  Category: [ ] mixed_sentiment  [ ] sarcasm  "
                         "[ ] genuinely_ambiguous  [ ] rating_inconsistency  [ ] other")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Qualitative error analysis on misclassified reviews'
    )
    parser.add_argument('--error_csv', required=True,
                        help='Path to *_errors.csv (from model_lr or model_bert)')
    parser.add_argument('--n', type=int, default=10,
                        help='Samples per true class (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save the report as a .txt file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    args = parser.parse_args()

    df = pd.read_csv(args.error_csv)
    print(f"Loaded {len(df):,} misclassified reviews from {args.error_csv}\n")

    print("Misclassification counts by true class:")
    print(df['true'].value_counts().to_string())
    print("\nConfusion breakdown  (true → predicted):")
    print(df.groupby(['true', 'predicted']).size().to_string())

    sample_df = sample_errors(df, n_per_class=args.n, seed=args.seed)
    report    = format_report(sample_df)

    print("\n" + report)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved -> {args.output}")


if __name__ == '__main__':
    main()
