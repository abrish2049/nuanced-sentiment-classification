# python all_models_score_viz.py [--max_len 128|256|512]
# requires the models to be run first and the results folder to be created

import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=None,
                    choices=[128, 256, 512],
                    help='Max length used during training. '
                         'If omitted, tries all_results_len512.json then all_results.json')
args = parser.parse_args()

# Resolve the all-models results file
max_len_tag = args.max_len or 512
results_dir = f'results/len{max_len_tag}'

if args.max_len is not None:
    results_file = os.path.join(results_dir, f'all_results_len{args.max_len}.json')
else:
    candidates = sorted(glob.glob('results/len*/all_results_len*.json'))
    if candidates:
        results_file = candidates[-1]
        results_dir  = os.path.dirname(results_file)
    else:
        raise FileNotFoundError(
            "No results JSON found. Run train_all_models.py first.")

# Resolve the BERT-specific results file
bert_results_file = os.path.join(results_dir, f'bert_len{max_len_tag}_results.json')

print(f"Loading all-model results from : {results_file}")
print(f"Loading BERT ablation results from: {bert_results_file}")

sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.3)

with open(results_file, 'r') as f:
    data = json.load(f)


# ------------------------------------------------------------------ #
# PLOT 1 — Accuracy vs Macro-F1 scatter (with_weighting, all models) #
# ------------------------------------------------------------------ #
accuracy    = []
f1_scores   = []
valid_models = []

for model in data:
    res = data[model]
    if 'with_weighting' in res:
        valid_models.append(model)
        accuracy.append(res['with_weighting']['test_accuracy'])
        f1_scores.append(res['with_weighting']['test_macro_f1'])

fig, ax = plt.subplots(figsize=(14, 10))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for i, model in enumerate(valid_models):
    ax.scatter(accuracy[i], f1_scores[i], s=500, alpha=0.7, c=colors[i],
               edgecolors='black', linewidth=2.5, zorder=3)
for i, model in enumerate(valid_models):
    ax.text(accuracy[i], f1_scores[i], model, fontsize=11, fontweight='bold',
            ha='left', va='top', rotation=-30,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

min_val = min(min(accuracy), min(f1_scores)) - 0.01
max_val = max(max(accuracy), max(f1_scores)) + 0.01
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, linewidth=2,
        label='Accuracy = F1', zorder=1)

ax.set_xlabel('Accuracy', fontsize=15, fontweight='bold', labelpad=10)
ax.set_ylabel('Macro F1 Score', fontsize=15, fontweight='bold', labelpad=10)
ax.set_title('Model Performance: Accuracy vs F1 Score', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.legend(fontsize=12, frameon=True, shadow=True, loc='lower right')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
scatter_path = os.path.join(results_dir, f'model_performance_comparison_len{max_len_tag}.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"Scatter plot saved as '{scatter_path}'")
plt.show()

# Print summary
print("\n" + "=" * 70)
print("                    PERFORMANCE SUMMARY")
print("=" * 70)
print(f"{'Model':<20} {'Accuracy':>12} {'Macro F1':>12} {'Difference':>12}")
print("-" * 70)
for model in valid_models:
    acc  = data[model]['with_weighting']['test_accuracy']
    f1   = data[model]['with_weighting']['test_macro_f1']
    diff = acc - f1
    print(f"{model:<20} {acc:>12.4f} {f1:>12.4f} {diff:>12.4f}")

best_model = max(valid_models, key=lambda m: data[m]['with_weighting']['test_macro_f1'])
print("=" * 70)
print(f"Best Model: {best_model} (F1: {data[best_model]['with_weighting']['test_macro_f1']:.4f})")
print("=" * 70)


# ------------------------------------------------------------------ #
# PLOT 2 — BERT ablation grouped bar chart                           #
# Variants: no_weighting, with_weighting, focal, last2_layers,       #
#           head_only                                                 #
# Metrics:  Accuracy, Macro-F1, Neutral F1                           #
# ------------------------------------------------------------------ #
if not os.path.isfile(bert_results_file):
    print(f"\n[skip] BERT ablation file not found: {bert_results_file}")
    print("       Run: python train_all_models.py --model bert")
else:
    with open(bert_results_file, 'r') as f:
        bert_data = json.load(f)

    variant_order  = ['no_weighting', 'with_weighting', 'focal', 'last2_layers', 'head_only']
    variant_labels = ['No weighting', 'With weighting', 'Focal (γ=2)', 'Last 2 layers', 'Head only']

    variants_present = [v for v in variant_order if v in bert_data]
    labels_present   = [variant_labels[variant_order.index(v)] for v in variants_present]

    accs        = [bert_data[v]['test_accuracy']              for v in variants_present]
    macro_f1s   = [bert_data[v]['test_macro_f1']              for v in variants_present]
    neutral_f1s = [bert_data[v]['per_class_f1']['neutral']    for v in variants_present]

    x  = np.arange(len(variants_present))
    w  = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(variants_present) * 2.2), 6))

    bars_acc  = ax.bar(x - w, accs,        w, label='Accuracy',   color='steelblue',  zorder=3)
    bars_f1   = ax.bar(x,     macro_f1s,   w, label='Macro-F1',   color='seagreen',   zorder=3)
    bars_nf1  = ax.bar(x + w, neutral_f1s, w, label='Neutral F1', color='tomato',     zorder=3)

    # value labels on top of each bar
    for bars in (bars_acc, bars_f1, bars_nf1):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, fontsize=11)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'BERT Ablation — len={max_len_tag}', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    plt.tight_layout()
    bert_chart_path = os.path.join(results_dir, f'bert_ablation_len{max_len_tag}.png')
    plt.savefig(bert_chart_path, dpi=300, bbox_inches='tight')
    print(f"\nBERT ablation chart saved as '{bert_chart_path}'")
    plt.show()

    # Print BERT ablation summary table
    print("\n" + "=" * 70)
    print(f"             BERT ABLATION SUMMARY  (len={max_len_tag})")
    print("=" * 70)
    print(f"{'Variant':<20} {'Accuracy':>12} {'Macro-F1':>12} {'Neutral F1':>12}")
    print("-" * 70)
    for v, label in zip(variants_present, labels_present):
        acc  = bert_data[v]['test_accuracy']
        f1   = bert_data[v]['test_macro_f1']
        nf1  = bert_data[v]['per_class_f1']['neutral']
        print(f"{label:<20} {acc:>12.4f} {f1:>12.4f} {nf1:>12.4f}")
    print("=" * 70)