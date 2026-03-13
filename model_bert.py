import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

from data_handler import CLASSES, RESULTS_DIR, device
from visualization import plot_confusion_matrix, plot_training_curves

BERT_MAX_LEN = 512
BERT_BATCH   = 16
BERT_LR      = 2e-5
BERT_EPOCHS  = 5
LABEL_MAP    = {'bad': 0, 'neutral': 1, 'good': 2}


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize BERTDataset instance.

        Parameters
        ----------
        texts : List[str]
            List of texts to be processed.
        labels : List[int]
            List of corresponding labels.
        tokenizer : BertTokenizer
            Pre-trained BERT tokenizer.
        max_len : int
            Maximum length of the input sequence.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to be retrieved.

        Returns
        -------
        tuple
            A tuple containing the input_ids, attention_mask, and label of the sample.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length = self.max_len,
            padding    = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        
        return (
            encoding['input_ids'].squeeze(0),  # shape: (max_len,)
            encoding['attention_mask'].squeeze(0),  # shape: (max_len,)
            torch.tensor(self.labels[idx], dtype=torch.long)  # shape: ()
        )


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3, freeze_layers=0):
        """
        Initialize BERTClassifier instance.

        Parameters
        ----------
        num_classes : int, default=3
            Number of classes in the classification task.
        dropout : float, default=0.3
            Dropout probability for the classifier head.
        freeze_layers : int, default=0
            Number of layers to freeze from the pre-trained BERT model.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        # BERT-base hidden size is always 768
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BERTClassifier.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of shape (batch_size, max_len) containing token IDs.
        attention_mask : torch.Tensor
            Tensor of shape (batch_size, max_len) containing attention masks.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_classes) containing class logits.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask = attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

def run_bert(train_df, val_df, test_df, weight_tensor):
    """
    Train and evaluate a BERT model on the given datasets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        DataFrame containing the training data.
    val_df : pandas.DataFrame
        DataFrame containing the validation data.
    test_df : pandas.DataFrame
        DataFrame containing the test data.
    weight_tensor : torch.Tensor or None
        Tensor of shape (num_classes,) containing class weights.

    Returns
    -------
    dict
        A dictionary containing the results of the experiment.
    """
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

    train_loader = DataLoader(train_dataset, batch_size = BERT_BATCH, shuffle = True,  num_workers = 4)
    val_loader   = DataLoader(val_dataset,   batch_size = BERT_BATCH, shuffle = False, num_workers = 4)
    test_loader  = DataLoader(test_dataset,  batch_size = BERT_BATCH, shuffle = False, num_workers = 4)

    all_results = {}

    for tag, cw, freeze in [
        ('no_weighting',   None,          0),
        ('with_weighting', weight_tensor, 0)
    ]:
        print("\n" + "=" * 60)
        print(f"BERT -- {tag}")
        print("=" * 60)

        model     = BERTClassifier(num_classes=  3, dropout = 0.3, freeze_layers = freeze).to(device)
        criterion = nn.CrossEntropyLoss(weight = cw) if cw is not None else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr = BERT_LR, weight_decay = 0.01)

        total_steps  = len(train_loader) * BERT_EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = warmup_steps,
            num_training_steps = total_steps
        )

        best_val_f1 = 0
        ckpt        = os.path.join(RESULTS_DIR, f'bert_{tag}_best.pt')
        history     = {'train_loss': [], 'val_loss': [],
                       'train_acc':  [], 'val_acc':  [],
                       'train_f1':   [], 'val_f1':   []}

        t0 = time.time()

        for epoch in range(BERT_EPOCHS):
            model.train()
            tr_loss, tr_preds, tr_labels = 0.0, [], []

            for input_ids, attention_mask, labels in tqdm(
                train_loader, desc = f"  Epoch {epoch+1}/{BERT_EPOCHS} Train", leave = False
            ):
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device)
                )

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

                optimizer.step()
                scheduler.step()

                tr_loss += loss.item()
                tr_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                tr_labels.extend(labels.cpu().numpy())

            tr_loss /= len(train_loader)
            tr_acc   = accuracy_score(tr_labels, tr_preds)
            tr_f1    = f1_score(tr_labels, tr_preds, average='macro')


            model.eval()
            vl_loss, vl_preds, vl_labels = 0.0, [], []

            with torch.no_grad():
                for input_ids, attention_mask, labels in tqdm(
                    val_loader, desc = f"  Epoch {epoch+1}/{BERT_EPOCHS} Val  ", leave = False
                ):
                    input_ids, attention_mask, labels = (
                        input_ids.to(device),
                        attention_mask.to(device),
                        labels.to(device)
                    )
                    logits     = model(input_ids, attention_mask)
                    vl_loss   += criterion(logits, labels).item()
                    vl_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    vl_labels.extend(labels.cpu().numpy())

            vl_loss /= len(val_loader)
            vl_acc   = accuracy_score(vl_labels, vl_preds)
            vl_f1    = f1_score(vl_labels, vl_preds, average = 'macro')


            history['train_loss'].append(tr_loss)
            history['val_loss'].append(vl_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(vl_acc)
            history['train_f1'].append(tr_f1)
            history['val_f1'].append(vl_f1)

            print(f"  Epoch {epoch+1:>2}/{BERT_EPOCHS} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
                  f"Val Loss {vl_loss:.4f} Acc {vl_acc:.4f} F1 {vl_f1:.4f}")

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                torch.save(model.state_dict(), ckpt)
                print(f"    New best model saved (Val F1 = {vl_f1:.4f})")

        elapsed = (time.time() - t0) / 60

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        ts_loss, ts_preds, ts_labels = 0.0, [], []

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(
                test_loader, desc="  Test", leave = False
            ):
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device)
                )
                logits    = model(input_ids, attention_mask)
                ts_loss  += criterion(logits, labels).item()
                ts_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                ts_labels.extend(labels.cpu().numpy())

        ts_loss /= len(test_loader)
        ts_acc   = accuracy_score(ts_labels, ts_preds)
        ts_f1    = f1_score(ts_labels, ts_preds, average = 'macro')

        print(f"\n  BERT ({tag}) Test → Acc: {ts_acc:.4f} | Macro-F1: {ts_f1:.4f}")
        print(classification_report(ts_labels, ts_preds,
                                    target_names = CLASSES, digits = 4))

        curve_path = os.path.join(RESULTS_DIR, f'bert_{tag}_curves.png')
        plot_training_curves(history, f"BERT ({tag})", curve_path)

        cm_path = os.path.join(RESULTS_DIR, f'bert_{tag}_confusion.png')
        plot_confusion_matrix(ts_labels, ts_preds, f"BERT ({tag})", cm_path)

        pcf1 = f1_score(ts_labels, ts_preds, average = None, labels = [0, 1, 2])
        all_results[tag] = {
            'test_accuracy': float(ts_acc),
            'test_macro_f1': float(ts_f1),
            'per_class_f1': {
                'bad':     float(pcf1[0]),
                'neutral': float(pcf1[1]),
                'good':    float(pcf1[2])
            },
            'training_time_minutes': round(elapsed, 2)
        }
    out = os.path.join(RESULTS_DIR, 'bert_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent = 2)
    print(f"\nResults saved to {out}")
    return all_results
