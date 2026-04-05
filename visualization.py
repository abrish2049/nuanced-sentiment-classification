"""
visualization.py
================
All plotting utilities for the sentiment analysis pipeline.

Functions
---------
plot_training_curves   — loss + macro-F1 curves per epoch
plot_confusion_matrix  — heatmap of true vs predicted labels
plot_performance       — grouped bar chart comparing all models
print_section          — pretty console section header

Usage:
  from visualization import (plot_training_curves, plot_confusion_matrix,
                              plot_performance, print_section)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_handler import CLASSES, RESULTS_DIR


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def plot_training_curves(history, tag, save_path):
    """Save a two-panel figure: Loss (left) and Macro-F1 (right).

    Parameters
    ----------
    history   : dict with keys train_loss, val_loss, train_f1, val_f1
    tag       : str  — used in figure title
    save_path : str  — full path for the output PNG
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {tag}", fontsize=13)

    # Loss
    axes[0].plot(epochs, history['train_loss'],
                 label='Train Loss', color='steelblue')
    axes[0].plot(epochs, history['val_loss'],
                 label='Val Loss',   color='tomato')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Macro-F1
    axes[1].plot(epochs, history['train_f1'],
                 label='Train Macro-F1', color='steelblue')
    axes[1].plot(epochs, history['val_f1'],
                 label='Val Macro-F1',   color='tomato')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro-F1')
    axes[1].set_title('Macro-F1')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves -> {save_path}")


def plot_confusion_matrix(labels, preds, tag, save_path):
    """Save a seaborn heatmap of the confusion matrix.

    Parameters
    ----------
    labels    : array-like of int  — ground-truth class indices
    preds     : array-like of int  — predicted class indices
    tag       : str                — figure title
    save_path : str                — full path for the output PNG
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(tag)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def plot_performance(results_by_model, save_path):
    """Grouped bar chart: Accuracy, Macro-F1, and Neutral F1 per model variant.

    Parameters
    ----------
    results_by_model : dict
        { model_name: { 'no_weighting': {...}, 'with_weighting': {...} } }
    save_path : str
        Full path for the output PNG.
    """
    rows = []
    for model_name, res in results_by_model.items():
        for variant in ['no_weighting', 'with_weighting']:
            if variant not in res:
                continue
            r = res[variant]
            rows.append({
                'Model':      f"{model_name}\n({variant.replace('_', ' ')})",
                'Accuracy':   r.get('test_accuracy',              float('nan')),
                'Macro-F1':   r.get('test_macro_f1',              float('nan')),
                'Neutral F1': r.get('per_class_f1', {}).get('neutral', float('nan'))
            })

    if not rows:
        print("No data to plot for performance comparison.")
        return

    df  = pd.DataFrame(rows)
    x   = np.arange(len(df))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 5))

    ax.bar(x - w, df['Accuracy'],   w, label='Accuracy',   color='steelblue')
    ax.bar(x,     df['Macro-F1'],   w, label='Macro-F1',   color='seagreen')
    ax.bar(x + w, df['Neutral F1'], w, label='Neutral F1', color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], fontsize=8)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved performance chart -> {save_path}")


def plot_tfidf_features(lr_model, feature_names, save_path, top_n=15):
    """Horizontal bar chart of top TF-IDF features per class.

    Parameters
    ----------
    lr_model      : fitted LogisticRegression with coef_ attribute
    feature_names : array-like of str — from vectorizer.get_feature_names_out()
    save_path     : str — full path for the output PNG
    top_n         : int — number of top features to show per class
    """
    classes = CLASSES
    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5),
                             sharey=False)
    fig.suptitle('Top TF-IDF Features per Class (weighted LR)', fontsize=13)

    colors = {'bad': 'tomato', 'neutral': 'steelblue', 'good': 'seagreen'}

    for ax, cls in zip(axes, classes):
        i     = classes.index(cls)
        coefs = lr_model.coef_[i]
        top_i = np.argsort(coefs)[-top_n:][::-1]
        words = [feature_names[j] for j in top_i]
        vals  = coefs[top_i]

        ax.barh(range(top_n), vals[::-1], color=colors[cls], alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(words[::-1], fontsize=9)
        ax.set_title(f"'{cls}'", fontsize=11)
        ax.set_xlabel('Coefficient')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TF-IDF feature chart -> {save_path}")


def plot_rating_distribution(train_df, val_df, test_df, save_path):
    """Bar chart of raw 1–10 rating counts across the full dataset.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame — the three data splits
    save_path : str — full path for the output PNG
    """
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    counts  = full_df['rating'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
    ax.set_xticks(range(1, 11))
    ax.set_xlabel('Rating (1–10)')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('IMDb Rating Distribution')
    ax.grid(axis='y', alpha=0.3)
    max_count = counts.max()
    for x, y in zip(counts.index, counts.values):
        ax.text(x, y + max_count * 0.01, f'{y:,}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rating distribution -> {save_path}")


def print_and_save_comparison(all_model_results, csv_path=None):
    """Print a comparison table to stdout and optionally save as CSV.

    Parameters
    ----------
    all_model_results : dict  — same structure as plot_performance
    csv_path          : str or None — if given, also writes a CSV
    """
    print_section("FINAL MODEL COMPARISON")
    rows = []
    for model_name, res in all_model_results.items():
        for variant in ['no_weighting', 'with_weighting']:
            if variant not in res:
                continue
            r = res[variant]
            rows.append({
                'Model':      f"{model_name} ({variant.replace('_', ' ')})",
                'Accuracy':   r.get('test_accuracy',              float('nan')),
                'Macro-F1':   r.get('test_macro_f1',              float('nan')),
                'Neutral F1': r.get('per_class_f1', {}).get('neutral', float('nan'))
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format='{:.4f}'.format))

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved -> {csv_path}")