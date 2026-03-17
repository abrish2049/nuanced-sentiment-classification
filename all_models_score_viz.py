# For generating a png file of all 5 models perfomrance acc vs F1 
# python all_models_score_viz.py
# requires the models to berun first and the results folder to be created 
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.3)

with open('results/all_results.json', 'r') as f:
    data = json.load(f)

models = list(data.keys())
accuracy = [data[model]['with_weighting']['test_accuracy'] for model in models]
f1_score = [data[model]['with_weighting']['test_macro_f1'] for model in models]

fig, ax = plt.subplots(figsize=(14, 10))

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

# Scatter plot
for i, model in enumerate(models):
    ax.scatter(accuracy[i], f1_score[i], s=500, alpha=0.7, c=colors[i], 
               edgecolors='black', linewidth=2.5, zorder=3)
for i, model in enumerate(models):
    ax.text(accuracy[i], f1_score[i], model, fontsize=11, fontweight='bold',
            ha='left', va='top', rotation=-30, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

min_val = min(min(accuracy), min(f1_score)) - 0.01
max_val = max(max(accuracy), max(f1_score)) + 0.01
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, linewidth=2, 
        label='Accuracy = F1', zorder=1)

# Customize-able plot
ax.set_xlabel('Accuracy', fontsize=15, fontweight='bold', labelpad=10)
ax.set_ylabel('Macro F1 Score', fontsize=15, fontweight='bold', labelpad=10)
ax.set_title('Model Performance: Accuracy vs F1 Score', fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
ax.legend(fontsize=12, frameon=True, shadow=True, loc='lower right')


ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'model_performance_comparison.png'")
plt.show()

# Print summary
print("\n" + "="*70)
print("                    PERFORMANCE SUMMARY")
print("="*70)
print(f"{'Model':<20} {'Accuracy':>12} {'Macro F1':>12} {'Difference':>12}")
print("-"*70)
for model in models:
    acc = data[model]['with_weighting']['test_accuracy']
    f1 = data[model]['with_weighting']['test_macro_f1']
    diff = acc - f1
    print(f"{model:<20} {acc:>12.4f} {f1:>12.4f} {diff:>12.4f}")


## Additional summary 
best_model = max(models, key=lambda m: data[m]['with_weighting']['test_macro_f1'])
print("="*70)
print(f"Best Model: {best_model} (F1: {data[best_model]['with_weighting']['test_macro_f1']:.4f})")
print("="*70)
