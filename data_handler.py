"""
data_handler.py
===============
Centralised data loading, splitting, and vocabulary utilities.

Handles:
  - Loading and cleaning Data.csv
  - Sentiment label assignment (two schemes)
  - Stratified train / val / test split — with a cache check to avoid
    re-splitting if the CSVs already exist on disk
  - Class-weight computation
  - Vocabulary building, GloVe loading, and DataLoader creation
  - Re-exports SentimentDataset so the rest of the pipeline only needs
    to import from this module

Usage:
  from data_handler import load_and_split_data, compute_weights, \
                           build_vocab, create_dataloaders, load_glove_embeddings
"""

import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------ #
# GLOBAL CONFIG (shared by all modules via this import)              #
# ------------------------------------------------------------------ #
CLASSES      = ['bad', 'neutral', 'good']
LABEL_MAP    = {'bad': 0, 'neutral': 1, 'good': 2}
RANDOM_SEED  = 42
DATA_CSV     = 'Data.csv'
RESULTS_DIR  = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Split output paths — single source of truth
TRAIN_CSV = 'train_expanded.csv'
VAL_CSV   = 'val_expanded.csv'
TEST_CSV  = 'test_expanded.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================== #
# SECTION 1 — LABEL ASSIGNMENT                                       #
# ================================================================== #
def assign_sentiment(rating, scheme='default'):
    """Map a numeric rating to a sentiment label.

    Schemes
    -------
    default      : 1-4 → bad | 5-6 → neutral | 7-10 → good
    wide_neutral : 1-3 → bad | 4-7 → neutral | 8-10 → good
    """
    if scheme == 'wide_neutral':
        if rating <= 3:   return 'bad'
        elif rating <= 7: return 'neutral'
        else:             return 'good'
    elif scheme == 'narrow_neutral':
        if rating <= 5:   return 'bad'
        elif rating == 6: return 'neutral'
        else:             return 'good'
    else:
        if rating <= 4:   return 'bad'
        elif rating <= 6: return 'neutral'
        else:             return 'good'

# ================================================================== #
# SECTION 2 — LOAD & SPLIT (with cache guard)                        #
# ================================================================== #
def _splits_exist():
    """Return True only when all three split CSVs are present on disk."""
    return all(os.path.isfile(p) for p in [TRAIN_CSV, VAL_CSV, TEST_CSV])


def load_and_split_data(scheme='default', force_resplit=False):
    """Load Data.csv, apply label mapping, stratified split, save CSVs.

    Parameters
    ----------
    scheme : str
        'default' or 'wide_neutral' (see assign_sentiment).
    force_resplit : bool
        If True, ignore cached CSVs and re-split from scratch.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    # ---- cache hit ------------------------------------------------ #
    if _splits_exist() and not force_resplit:
        print("[data_handler] Split CSVs already exist — loading from disk.")
        print(f"  Delete {TRAIN_CSV} / {VAL_CSV} / {TEST_CSV} or pass "
              "force_resplit=True to regenerate.")
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)
        test_df  = pd.read_csv(TEST_CSV)
        _print_split_summary(train_df, val_df, test_df)
        return train_df, val_df, test_df

    # ---- fresh split ---------------------------------------------- #
    print("[data_handler] Generating fresh train / val / test split …")
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=['review', 'rating'])
    df['rating']    = df['rating'].astype(int)
    df['sentiment'] = df['rating'].apply(assign_sentiment, scheme=scheme)

    print(f"  Total reviews : {len(df):,}")
    print(f"  Label scheme  : {scheme}")
    print("  Class distribution:")
    counts = df['sentiment'].value_counts()
    for cls in CLASSES:
        pct = counts.get(cls, 0) / len(df) * 100
        print(f"    {cls:>8}: {counts.get(cls, 0):>6,}  ({pct:.1f}%)")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df['sentiment']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED,
        stratify=temp_df['sentiment']
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,     index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    print(f"  Saved {TRAIN_CSV}, {VAL_CSV}, {TEST_CSV}")

    _print_split_summary(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _print_split_summary(train_df, val_df, test_df):
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Test: {len(test_df):,}")


# ================================================================== #
# SECTION 3 — CLASS WEIGHTS                                          #
# ================================================================== #
def compute_weights(train_df):
    """Compute inverse-frequency class weights for the training set.

    Returns
    -------
    weight_dict   : dict  {class_name: float}
    weight_tensor : torch.Tensor  shape (3,), on `device`
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(CLASSES),
        y=train_df['sentiment']
    )
    weight_dict   = dict(zip(CLASSES, weights))
    weight_tensor = torch.tensor(
        [weight_dict['bad'], weight_dict['neutral'], weight_dict['good']],
        dtype=torch.float
    ).to(device)
    print(f"[data_handler] Class weights [bad, neutral, good]: "
          f"{weight_tensor.cpu().numpy().round(4)}")
    return weight_dict, weight_tensor


# ================================================================== #
# SECTION 4 — DATASET & VOCAB (from dataset.py, consolidated here)  #
# ================================================================== #
class SentimentDataset(Dataset):
    """Token-index dataset for TextCNN / BiLSTM models."""

    def __init__(self, texts, labels, vocab, max_len=512):
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text    = self.texts[idx]
        label   = self.labels[idx]
        tokens  = text.lower().split()[:self.max_len]
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        # Pad
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return (torch.tensor(indices, dtype=torch.long),
                torch.tensor(label,   dtype=torch.long))


def build_vocab(texts, min_freq=2, max_vocab_size=50000):
    """Build a word → index vocabulary from a list of texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)

    print(f"[data_handler] Vocabulary size: {len(vocab):,}")
    return vocab


def load_glove_embeddings(glove_path, vocab, embed_dim=300):
    """Load pretrained GloVe vectors for words present in *vocab*.

    Returns
    -------
    torch.FloatTensor  shape (len(vocab), embed_dim)
    """
    print(f"[data_handler] Loading GloVe from {glove_path} …")
    embeddings      = np.random.randn(len(vocab), embed_dim) * 0.01
    embeddings[0]   = 0  # PAD → zero vector
    found           = 0

    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split()
            word  = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"[data_handler] Found {found}/{len(vocab)} words in GloVe")
    return torch.FloatTensor(embeddings)


def create_dataloaders(train_df, val_df, test_df, vocab,
                       batch_size=32, max_len=512):
    """Wrap the three split DataFrames in PyTorch DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    def _make(df):
        return SentimentDataset(
            df['review'].values,
            df['sentiment'].map(LABEL_MAP).values,
            vocab, max_len
        )

    train_loader = DataLoader(_make(train_df), batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(_make(val_df),   batch_size=batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(_make(test_df),  batch_size=batch_size,
                              shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader
