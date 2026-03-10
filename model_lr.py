import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_handler import CLASSES, RANDOM_SEED, RESULTS_DIR
from visualization import plot_confusion_matrix


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

    for tag, cw in [('no_weighting', None), ('with_weighting', weight_dict)]:
        print(f"\n-- LR ({tag}) --")
        lr = LogisticRegression(
            C=1.0, solver='lbfgs',
            max_iter=1000, random_state=RANDOM_SEED, class_weight=cw
        )
        lr.fit(X_train, y_train)
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
    with open('logistic_regression_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results
