import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from data_handler import RESULTS_DIR, device, build_vocab, create_dataloaders
from visualization import plot_confusion_matrix, plot_training_curves
from training_utils import run_neural_experiment

BATCH_SIZE     = 64
MAX_LEN        = 512
EMBED_DIM      = 300
DROPOUT        = 0.5
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 15
MAX_VOCAB_SIZE = 50000
MIN_FREQ       = 2
CNN_NUM_FILTERS  = 100
CNN_FILTER_SIZES = [3, 4, 5]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, filter_sizes=(3, 4, 5),
                 dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.convs   = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            pooled.append(F.max_pool1d(c, c.size(2)).squeeze(2))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def run_textcnn(train_df, val_df, test_df, weight_tensor):
    print("\n" + "=" * 60)
    print("TextCNN")
    print("=" * 60)

    vocab = build_vocab(train_df['review'].values,
                        min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)

    # glove_embeddings = load_glove_embeddings('glove.6B.300d.txt', vocab, EMBED_DIM)
    glove_embeddings = None
    print("Using randomly initialized embeddings (no GloVe)")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, vocab,
        batch_size=BATCH_SIZE, max_len=MAX_LEN
    )

    all_results = {}

    for tag, cw in [('no_weighting', None), ('with_weighting', weight_tensor)]:
        print_section(f"TextCNN -- {tag}")
        model = TextCNN(
            vocab_size=len(vocab), embed_dim=EMBED_DIM,
            num_classes=3, num_filters=CNN_NUM_FILTERS,
            filter_sizes=CNN_FILTER_SIZES, dropout=DROPOUT,
            pretrained_embeddings=glove_embeddings
        ).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        criterion = (nn.CrossEntropyLoss(weight=cw)
                     if cw is not None else nn.CrossEntropyLoss())
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        t0  = time.time()
        acc, mf1, preds, labels, history = run_neural_experiment(
            f"textcnn_{tag}", model,
            train_loader, val_loader, test_loader,
            criterion, optimizer
        )
        elapsed = (time.time() - t0) / 60

        curve_path = os.path.join(RESULTS_DIR, f'textcnn_{tag}_curves.png')
        plot_training_curves(history, f"TextCNN ({tag})", curve_path)

        cm_path = os.path.join(RESULTS_DIR, f'textcnn_{tag}_confusion.png')
        plot_confusion_matrix(labels, preds, f"TextCNN ({tag})", cm_path)

        pcf1 = f1_score(labels, preds, average=None, labels=[0, 1, 2])
        all_results[tag] = {
            'test_accuracy': float(acc),
            'test_macro_f1': float(mf1),
            'per_class_f1': {
                'bad':     float(pcf1[0]),
                'neutral': float(pcf1[1]),
                'good':    float(pcf1[2])
            },
            'training_time_minutes': round(elapsed, 2)
        }

    out = os.path.join(RESULTS_DIR, 'textcnn_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    with open('textcnn_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
