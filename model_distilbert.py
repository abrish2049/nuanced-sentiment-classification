import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from data_handler import CLASSES, RESULTS_DIR, device
from visualization import plot_confusion_matrix, plot_training_curves

BERT_MAX_LEN = 512
BERT_BATCH   = 16
BERT_LR      = 2e-5
BERT_EPOCHS  = 5
LABEL_MAP    = {'bad': 0, 'neutral': 1, 'good': 2}


class DistilBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),       # (max_len,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # (max_len,)
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, freeze_bert=False):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        hidden_size = self.distilbert.config.hidden_size  # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs  = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]   # (batch, hidden_size)
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)           # (batch, num_classes)
        return logits


def _train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


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
        criterion = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None) \
                    if cw is not None else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

        total_steps  = len(train_loader) * BERT_EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        ckpt        = os.path.join(RESULTS_DIR, f'distilbert_{tag}_best.pt')
        history     = {
            'train_loss': [], 'val_loss': [],
            'train_acc':  [], 'val_acc':  [],
            'train_f1':   [], 'val_f1':   []
        }

        t0 = time.time()

        for epoch in range(BERT_EPOCHS):
            print(f"\n  Epoch {epoch + 1}/{BERT_EPOCHS}")

            # --- Train ---
            tr_loss, tr_acc, tr_f1 = _train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )

            # --- Validate ---
            vl_loss, vl_acc, vl_f1, _, _ = _eval_epoch(
                model, val_loader, criterion, device
            )

            # --- Log history ---
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)
            history['train_f1'].append(tr_f1)
            history['val_f1'].append(vl_f1)

            print(f"  train loss={tr_loss:.4f}  acc={tr_acc:.4f}  f1={tr_f1:.4f}")
            print(f"  val   loss={vl_loss:.4f}  acc={vl_acc:.4f}  f1={vl_f1:.4f}")

            # --- Save best checkpoint ---
            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                torch.save(model.state_dict(), ckpt)
                print(f"  ✓ New best val F1={best_val_f1:.4f} — checkpoint saved")

        elapsed = (time.time() - t0) / 60
        print(f"\n  Training time: {elapsed:.1f} min")

        # --- Load best checkpoint and evaluate on test set ---
        model.load_state_dict(torch.load(ckpt, map_location=device))
        te_loss, te_acc, te_f1, te_preds, te_labels = _eval_epoch(
            model, test_loader, criterion, device
        )
        print(f"\n  Test  loss={te_loss:.4f}  acc={te_acc:.4f}  f1={te_f1:.4f}")
        print(classification_report(te_labels, te_preds, target_names=CLASSES, zero_division=0))

        # --- Plots ---
        plot_training_curves(
            history,
            title=f'DistilBERT ({tag})',
            save_path=os.path.join(RESULTS_DIR, f'distilbert_{tag}_curves.png')
        )
        plot_confusion_matrix(
            te_labels, te_preds,
            classes=CLASSES,
            title=f'DistilBERT ({tag}) — Confusion Matrix',
            save_path=os.path.join(RESULTS_DIR, f'distilbert_{tag}_confusion.png')
        )

        # --- Store results ---
        all_results[tag] = {
            'test_accuracy':    te_acc,
            'test_macro_f1':    te_f1,
            'test_loss':        te_loss,
            'training_minutes': elapsed,
            'best_val_f1':      best_val_f1,
            'history':          history,
            'classification_report': classification_report(
                te_labels, te_preds,
                target_names=CLASSES,
                output_dict=True,
                zero_division=0
            )
        }

    out = os.path.join(RESULTS_DIR, 'distilbert_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    return all_results