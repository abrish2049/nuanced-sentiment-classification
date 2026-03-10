import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from data_handler import CLASSES, RESULTS_DIR, device
from visualization import plot_confusion_matrix, plot_training_curves

BERT_MAX_LEN = 512
BERT_BATCH   = 16
BERT_LR      = 2e-5
BERT_EPOCHS  = 5
LABEL_MAP    = {'bad': 0, 'neutral': 1, 'good': 2}


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        # TODO: implement
        raise NotImplementedError

    def __len__(self):
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: implement
        raise NotImplementedError


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, freeze_layers=0):
        super().__init__()
        # TODO: load BertModel, optionally freeze first N layers,
        #       define dropout and classifier head
        raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        # TODO: implement forward pass
        raise NotImplementedError


def run_bert(train_df, val_df, test_df, weight_tensor):
    print("\n" + "=" * 60)
    print("BERT-base")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = BERTDataset(
        train_df['review'].tolist(),
        train_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )
    val_dataset = BERTDataset(
        val_df['review'].tolist(),
        val_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )
    test_dataset = BERTDataset(
        test_df['review'].tolist(),
        test_df['sentiment'].map(LABEL_MAP).tolist(),
        tokenizer, BERT_MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BERT_BATCH, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=BERT_BATCH, shuffle=False, num_workers=4)

    all_results = {}

    for tag, cw, freeze in [
        ('no_weighting',   None,          0),
        ('with_weighting', weight_tensor, 0)
    ]:
        print("\n" + "=" * 60)
        print(f"BERT -- {tag}")
        print("=" * 60)

        model     = BERTClassifier(num_classes=3, dropout=0.3, freeze_layers=freeze).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw) if cw is not None else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

        total_steps  = len(train_loader) * BERT_EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_f1 = 0
        ckpt        = os.path.join(RESULTS_DIR, f'bert_{tag}_best.pt')
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

    out = os.path.join(RESULTS_DIR, 'bert_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    return all_results
