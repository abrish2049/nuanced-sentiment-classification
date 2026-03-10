import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup

from data_handler import CLASSES, RESULTS_DIR, device
from visualization import plot_confusion_matrix, plot_training_curves

BERT_MAX_LEN = 512
BERT_BATCH   = 16
BERT_LR      = 2e-5
BERT_EPOCHS  = 5
LABEL_MAP    = {'bad': 0, 'neutral': 1, 'good': 2}


class DistilBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        # TODO: implement
        raise NotImplementedError

    def __len__(self):
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: implement
        raise NotImplementedError


class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, freeze_bert=False):
        super().__init__()
        # TODO: load DistilBertModel, define dropout and classifier head
        raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        # TODO: implement forward pass
        raise NotImplementedError


def run_distilbert(train_df, val_df, test_df, weight_tensor):
    print("\n" + "=" * 60)
    print("DistilBERT")
    print("=" * 60)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = DistilBERTDataset(
        train_df['review'].tolist(),
        train_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )
    val_dataset = DistilBERTDataset(
        val_df['review'].tolist(),
        val_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )
    test_dataset = DistilBERTDataset(
        test_df['review'].tolist(),
        test_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BERT_BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BERT_BATCH, shuffle=False, num_workers=4)

    all_results = {}

    for tag, cw, freeze in [
        ('no_weighting',   None,          False),
        ('with_weighting', weight_tensor, False)
    ]:
        print("\n" + "=" * 60)
        print(f"DistilBERT -- {tag}")
        print("=" * 60)

        model     = DistilBERTClassifier(num_classes=3, dropout=0.3, freeze_bert=freeze).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw) if cw is not None else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

        total_steps  = len(train_loader) * BERT_EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_f1 = 0
        ckpt        = os.path.join(RESULTS_DIR, f'distilbert_{tag}_best.pt')
        history     = {'train_loss': [], 'val_loss': [],
                       'train_acc':  [], 'val_acc':  [],
                       'train_f1':   [], 'val_f1':   []}

        t0 = time.time()

        for epoch in range(BERT_EPOCHS):
            # TODO: implement train loop
            pass

            # TODO: implement val loop
            pass

            # TODO: log history, save best checkpoint

        elapsed = (time.time() - t0) / 60

        # TODO: load best checkpoint and run test evaluation
        # TODO: call plot_training_curves and plot_confusion_matrix
        # TODO: populate all_results[tag]

    out = os.path.join(RESULTS_DIR, 'distilbert_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results
