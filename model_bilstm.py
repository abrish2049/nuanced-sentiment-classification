import json
import os
import time

import torch
import torch.nn as nn
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
HIDDEN_DIM     = 256


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                       # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(emb)

        # hidden shape: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]                  # last layer, forward direction
        backward_hidden = hidden[-1]                 # last layer, backward direction

        hidden_cat = torch.cat((forward_hidden, backward_hidden), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)


def run_bilstm(train_df, val_df, test_df, weight_tensor, max_len=512):
    print("\n" + "=" * 60)
    print("BiLSTM")
    print("=" * 60)

    vocab = build_vocab(train_df['review'].values,
                        min_freq=MIN_FREQ, max_vocab_size=MAX_VOCAB_SIZE)

    glove_embeddings = None

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, vocab,
        batch_size=BATCH_SIZE, max_len=max_len
    )
    all_results = {}

    for tag, cw in [('no_weighting', None), ('with_weighting', weight_tensor)]:
        print("\n" + "=" * 60)
        print(f"BiLSTM -- {tag}")
        print("=" * 60)

        run_tag = f'bilstm_len{max_len}_{tag}'

        model = BiLSTM(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=3,
            num_layers=2,
            dropout=DROPOUT,
            pretrained_embeddings=glove_embeddings
        ).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        criterion = (nn.CrossEntropyLoss(weight=cw)
                     if cw is not None else nn.CrossEntropyLoss())
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        t0 = time.time()
        acc, mf1, preds, labels, history = run_neural_experiment(
            run_tag, model,                          # ← was f"bilstm_{tag}"
            train_loader, val_loader, test_loader,
            criterion, optimizer
        )
        elapsed = (time.time() - t0) / 60

        curve_path = os.path.join(RESULTS_DIR, f'{run_tag}_curves.png')
        plot_training_curves(history, f"BiLSTM ({tag}, len={max_len})", curve_path)

        cm_path = os.path.join(RESULTS_DIR, f'{run_tag}_confusion.png')
        plot_confusion_matrix(labels, preds, f"BiLSTM ({tag}, len={max_len})", cm_path)

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

    out = os.path.join(RESULTS_DIR, f'bilstm_len{max_len}_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results
