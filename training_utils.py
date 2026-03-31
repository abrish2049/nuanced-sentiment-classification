"""
training_utils.py
=================
Shared neural training helpers used by all model modules.

Functions
---------
train_epoch          — one epoch of training, returns loss/acc/f1
evaluate             — one pass of eval, returns loss/acc/f1/preds/labels
run_neural_experiment — full train → checkpoint → test loop for nn.Module models

These are model-agnostic: any model whose forward() takes a batch of
token-index tensors and returns class logits can use train_epoch /
evaluate directly. BERT-style models (two-input forward) should
replicate the loop inline (as done in model_distilbert.py / model_bert.py).

Dependencies:
  data_handler.py  — RESULTS_DIR, device
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_handler import RESULTS_DIR, device, CLASSES

NUM_EPOCHS = 15   # default; callers may override


def save_history_csv(history, tag, test_loss=None, test_acc=None, test_f1=None):
    """Save per-epoch training history (and optional test metrics) to CSV.

    Parameters
    ----------
    history   : dict with keys train_loss, val_loss, train_acc, val_acc, train_f1, val_f1
    tag       : str — used in the filename, e.g. 'bert_len512_focal'
    test_loss : float or None — final test loss (appended as last row)
    test_acc  : float or None — final test accuracy
    test_f1   : float or None — final test macro-F1
    """
    safe_tag = tag.lower().replace(" ", "_").replace("(", "").replace(")", "")
    rows = []
    for i in range(len(history['train_loss'])):
        rows.append({
            'epoch':      i + 1,
            'train_loss': history['train_loss'][i],
            'train_acc':  history['train_acc'][i],
            'train_f1':   history['train_f1'][i],
            'val_loss':   history['val_loss'][i],
            'val_acc':    history['val_acc'][i],
            'val_f1':     history['val_f1'][i],
        })
    if test_loss is not None:
        rows.append({
            'epoch':      'test',
            'train_loss': None,
            'train_acc':  None,
            'train_f1':   None,
            'val_loss':   test_loss,
            'val_acc':    test_acc,
            'val_f1':     test_f1,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f'{safe_tag}_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved training history -> {csv_path}")


def train_epoch(model, loader, criterion, optimizer):
    """Run one training epoch.

    Returns
    -------
    (avg_loss, accuracy, macro_f1)
    """
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for texts, labels in tqdm(loader, desc="  Train", leave=False):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average='macro')
    )


def evaluate(model, loader, criterion):
    """Run one evaluation pass (no gradient).

    Returns
    -------
    (avg_loss, accuracy, macro_f1, preds_list, labels_list)
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for texts, labels in tqdm(loader, desc="  Eval ", leave=False):
            texts, labels = texts.to(device), labels.to(device)
            outputs    = model(texts)
            total_loss += criterion(outputs, labels).item()
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average='macro'),
        all_preds,
        all_labels
    )


def run_neural_experiment(tag, model, train_loader, val_loader, test_loader,
                          criterion, optimizer, num_epochs=NUM_EPOCHS):
    """Generic train / validate / test loop for single-input nn.Module models.

    Saves the best checkpoint (by Val Macro-F1) to RESULTS_DIR.

    Parameters
    ----------
    tag          : str   — used for checkpoint filename and print headers
    model        : nn.Module whose forward(x) returns logits
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    criterion    : loss function
    optimizer    : torch optimizer
    num_epochs   : int

    Returns
    -------
    (test_acc, test_macro_f1, test_preds, test_labels, history_dict)
    """
    print(f"\n  Training: {tag}")
    best_val_f1 = 0.0
    safe_tag    = tag.lower().replace(" ", "_").replace("(", "").replace(")", "")
    ckpt        = os.path.join(RESULTS_DIR, f'{safe_tag}_best.pt')

    history = {
        'train_loss': [], 'val_loss':  [],
        'train_acc':  [], 'val_acc':   [],
        'train_f1':   [], 'val_f1':    []
    }

    for epoch in range(num_epochs):
        tr_loss, tr_acc, tr_f1         = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _  = evaluate(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['train_f1'].append(tr_f1)
        history['val_f1'].append(vl_f1)

        print(f"  Epoch {epoch+1:>2}/{num_epochs} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f} F1 {vl_f1:.4f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save(model.state_dict(), ckpt)
            print(f"    New best model saved (Val F1={vl_f1:.4f})")

    model.load_state_dict(torch.load(ckpt))
    ts_loss, ts_acc, ts_f1, ts_preds, ts_labels = evaluate(
        model, test_loader, criterion
    )

    print(f"\n  {tag} Test Results → Acc: {ts_acc:.4f} | Macro-F1: {ts_f1:.4f}")
    print(classification_report(ts_labels, ts_preds,
                                target_names=CLASSES, digits=4))

    save_history_csv(history, tag,
                     test_loss=ts_loss, test_acc=ts_acc, test_f1=ts_f1)

    return ts_acc, ts_f1, ts_preds, ts_labels, history