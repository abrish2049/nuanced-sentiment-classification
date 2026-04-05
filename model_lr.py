import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_handler import CLASSES, RANDOM_SEED, RESULTS_DIR
from visualization import plot_confusion_matrix, plot_tfidf_features


def run_logistic_regression(train_df, val_df, test_df, weight_dict):
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"Data check -> Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(test_df['sentiment'].value_counts())

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df['review'])
    X_test  = vectorizer.transform(test_df['review'])
    y_train = train_df['sentiment']
    y_test  = test_df['sentiment']

    all_results = {}
    lr_for_analysis = None
    
    for tag, cw in [('no_weighting', None), ('with_weighting', weight_dict)]:
        print(f"\n-- LR ({tag}) --")
        lr = LogisticRegression(
            C=1.0, solver='lbfgs',
            max_iter=1000, random_state=RANDOM_SEED, class_weight=cw
        )
        lr.fit(X_train, y_train)
        
        if tag == 'with_weighting':
            lr_for_analysis = lr
        
        preds = lr.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        mf1   = f1_score(y_test, preds, average='macro')
        pcf1  = f1_score(y_test, preds, average=None, labels=CLASSES)

        print(f"  Test Acc: {acc:.4f} | Macro-F1: {mf1:.4f}")
        print(classification_report(y_test, preds, target_names=CLASSES, digits=4))

        cm_path = os.path.join(RESULTS_DIR, f'lr_{tag}_confusion.png')
        plot_confusion_matrix(y_test, preds, f"LR ({tag})", cm_path)

        all_results[tag] = {
            'test_accuracy': float(acc),
            'test_macro_f1': float(mf1),
            'per_class_f1': {
                'bad':     float(pcf1[0]),
                'neutral': float(pcf1[1]),
                'good':    float(pcf1[2])
            }
        }

    # Feature analysis — top TF-IDF features per class (weighted model)
    print("\n-- Top TF-IDF features per class (weighted model) --")
    feature_names = vectorizer.get_feature_names_out()
    for i, cls in enumerate(CLASSES):
        coefs   = lr_for_analysis.coef_[i]
        top_pos = np.argsort(coefs)[-20:][::-1]
        print(f"\n  '{cls}' — most positive features:")
        print("  ", [feature_names[j] for j in top_pos])

    tfidf_chart_path = os.path.join(RESULTS_DIR, 'lr_tfidf_features.png')
    plot_tfidf_features(lr_for_analysis, feature_names, tfidf_chart_path)

    # Error analysis — look at misclassified examples, especially neutrals
    print("\n-- Error analysis (weighted model) --")
    error_df = pd.DataFrame({
        'review':    test_df['review'].values,
        'rating':    test_df['rating'].values,
        'true':      y_test.values,
        'predicted': lr_for_analysis.predict(X_test)
    })
    error_df['correct'] = error_df['true'] == error_df['predicted']
    errors = error_df[~error_df['correct']]
    errors.to_csv(os.path.join(RESULTS_DIR, 'lr_errors.csv'), index=False)
    neutral_errors = errors[errors['true'] == 'neutral']
    print(f"\n  Neutral misclassifications: {len(neutral_errors)}")
    print(neutral_errors['predicted'].value_counts().to_string())
    print(f"\n  Neutral errors by rating:")
    print(neutral_errors['rating'].value_counts().sort_index().to_string())

    # binary reference — drop neutral, run bad vs good only
    print("\n-- Binary reference (bad vs good, neutral dropped) --")
    bin_train = train_df[train_df['sentiment'] != 'neutral']
    bin_test  = test_df[test_df['sentiment']   != 'neutral']
    X_bt      = vectorizer.transform(bin_train['review'])
    X_bte     = vectorizer.transform(bin_test['review'])
    lr_bin    = LogisticRegression(C=1.0, solver='lbfgs',
                                   max_iter=1000, random_state=RANDOM_SEED)
    lr_bin.fit(X_bt, bin_train['sentiment'])
    bin_preds = lr_bin.predict(X_bte)
    bin_acc   = accuracy_score(bin_test['sentiment'], bin_preds)
    bin_f1    = f1_score(bin_test['sentiment'], bin_preds, average='macro')
    print(f"  Binary Acc: {bin_acc:.4f} | Binary Macro-F1: {bin_f1:.4f}")
    all_results['binary_reference'] = {
        'accuracy': float(bin_acc), 'macro_f1': float(bin_f1)
    }

    out = os.path.join(RESULTS_DIR, 'logistic_regression_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results
