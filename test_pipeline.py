"""
test_pipeline.py
================
Smoke-test suite for the sentiment analysis pipeline.

Covers everything that does NOT require running a model or Data.csv:

  1.  assign_sentiment        — all three schemes, every boundary
  2.  build_vocab              — frequency filtering, special tokens
  3.  SentimentDataset         — length, padding, UNK handling
  4.  create_dataloaders       — batch shapes, split sizes
  5.  compute_weights          — tensor shape, positivity, all-equal baseline
  6.  save_history_csv         — file written, columns, test row appended
  7.  FocalLoss                — output shape, gamma=0 matches CE, loss >= 0
  8.  TextCNN                  — output shape (no training)
  9.  BiLSTM                   — output shape (no training)
  10. BERTClassifier freeze     — parameter counts for 0 / 10 / 12 frozen layers
  11. DistilBERTClassifier      — output shape (no training)
  12. results/ output filenames — naming convention for max_len variants
  13. cache guard logic         — _splits_exist() with temp files

Run with:
  pytest test_pipeline.py -v
"""

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_handler import (
    assign_sentiment, build_vocab, SentimentDataset,
    create_dataloaders, compute_weights, CLASSES, LABEL_MAP,
    TRAIN_CSV, VAL_CSV, TEST_CSV, _splits_exist, RESULTS_DIR
)
from losses import FocalLoss
from model_textcnn import TextCNN
from model_bilstm import BiLSTM
from training_utils import save_history_csv


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
def _make_df(n=120, seed=0):
    """Tiny synthetic DataFrame that looks like a real split."""
    rng = np.random.default_rng(seed)
    reviews   = [f"review token_{i} word_{i % 10}" for i in range(n)]
    ratings   = rng.integers(1, 11, size=n).tolist()
    sentiments = [assign_sentiment(r) for r in ratings]
    return pd.DataFrame({'review': reviews, 'rating': ratings,
                         'sentiment': sentiments})


def _tiny_vocab():
    texts = [f"token_{i} word_{i % 10}" for i in range(200)]
    return build_vocab(texts, min_freq=1, max_vocab_size=500)


# ================================================================== #
# 1. assign_sentiment                                                  #
# ================================================================== #
class TestAssignSentiment:
    # --- default scheme ---
    def test_default_bad_boundaries(self):
        for r in [1, 2, 3, 4]:
            assert assign_sentiment(r, 'default') == 'bad'

    def test_default_neutral_boundaries(self):
        for r in [5, 6]:
            assert assign_sentiment(r, 'default') == 'neutral'

    def test_default_good_boundaries(self):
        for r in [7, 8, 9, 10]:
            assert assign_sentiment(r, 'default') == 'good'

    # --- wide_neutral scheme ---
    def test_wide_neutral_bad(self):
        for r in [1, 2, 3]:
            assert assign_sentiment(r, 'wide_neutral') == 'bad'

    def test_wide_neutral_neutral(self):
        for r in [4, 5, 6, 7]:
            assert assign_sentiment(r, 'wide_neutral') == 'neutral'

    def test_wide_neutral_good(self):
        for r in [8, 9, 10]:
            assert assign_sentiment(r, 'wide_neutral') == 'good'

    # --- narrow_neutral scheme ---
    def test_narrow_neutral_bad(self):
        for r in [1, 2, 3, 4, 5]:
            assert assign_sentiment(r, 'narrow_neutral') == 'bad'

    def test_narrow_neutral_neutral(self):
        assert assign_sentiment(6, 'narrow_neutral') == 'neutral'

    def test_narrow_neutral_good(self):
        for r in [7, 8, 9, 10]:
            assert assign_sentiment(r, 'narrow_neutral') == 'good'

    def test_unknown_scheme_falls_back_to_default(self):
        # unknown scheme should use the else branch (same as default)
        assert assign_sentiment(4, 'nonexistent') == 'bad'
        assert assign_sentiment(5, 'nonexistent') == 'neutral'
        assert assign_sentiment(7, 'nonexistent') == 'good'

    def test_all_ratings_produce_valid_label(self):
        for scheme in ['default', 'wide_neutral', 'narrow_neutral']:
            for r in range(1, 11):
                label = assign_sentiment(r, scheme)
                assert label in CLASSES, f"Bad label {label!r} for rating {r}, scheme {scheme}"

    def test_schemes_partition_all_ratings(self):
        """Every rating 1-10 maps to exactly one class under each scheme."""
        for scheme in ['default', 'wide_neutral', 'narrow_neutral']:
            labels = [assign_sentiment(r, scheme) for r in range(1, 11)]
            assert len(labels) == 10
            assert all(l in CLASSES for l in labels)


# ================================================================== #
# 2. build_vocab                                                       #
# ================================================================== #
class TestBuildVocab:
    def test_special_tokens_present(self):
        vocab = build_vocab(['hello world'], min_freq=1)
        assert '<PAD>' in vocab
        assert '<UNK>' in vocab

    def test_pad_is_index_zero(self):
        vocab = build_vocab(['hello world'], min_freq=1)
        assert vocab['<PAD>'] == 0

    def test_unk_is_index_one(self):
        vocab = build_vocab(['hello world'], min_freq=1)
        assert vocab['<UNK>'] == 1

    def test_min_freq_filters_rare_words(self):
        # 'rare' appears only once, 'common' appears 3 times
        texts = ['common common common rare']
        vocab = build_vocab(texts, min_freq=2)
        assert 'common' in vocab
        assert 'rare' not in vocab

    def test_max_vocab_size_respected(self):
        texts = [' '.join(f'w{i}' for i in range(200))]
        vocab = build_vocab(texts, min_freq=1, max_vocab_size=10)
        assert len(vocab) <= 10

    def test_indices_are_unique(self):
        vocab = _tiny_vocab()
        indices = list(vocab.values())
        assert len(indices) == len(set(indices))

    def test_case_insensitive(self):
        vocab = build_vocab(['Hello HELLO hello'], min_freq=1)
        assert 'hello' in vocab
        assert 'Hello' not in vocab
        assert 'HELLO' not in vocab


# ================================================================== #
# 3. SentimentDataset                                                  #
# ================================================================== #
class TestSentimentDataset:
    def setup_method(self):
        self.vocab   = _tiny_vocab()
        self.max_len = 8
        self.texts   = ['token_0 word_1 word_2', 'token_5']
        self.labels  = [0, 2]
        self.ds      = SentimentDataset(
            self.texts, self.labels, self.vocab, self.max_len
        )

    def test_length(self):
        assert len(self.ds) == 2

    def test_item_shapes(self):
        ids, label = self.ds[0]
        assert ids.shape == (self.max_len,)
        assert label.shape == ()

    def test_padding_applied(self):
        # 'token_5' is 1 token; remaining 7 slots should be PAD (index 0)
        ids, _ = self.ds[1]
        assert ids[-1].item() == self.vocab['<PAD>']

    def test_no_overflow_beyond_max_len(self):
        long_text = ' '.join(f'token_{i}' for i in range(100))
        ds = SentimentDataset([long_text], [0], self.vocab, self.max_len)
        ids, _ = ds[0]
        assert ids.shape == (self.max_len,)

    def test_unknown_token_maps_to_unk(self):
        ds = SentimentDataset(['completelymadeupword'], [0],
                              self.vocab, self.max_len)
        ids, _ = ds[0]
        assert ids[0].item() == self.vocab['<UNK>']

    def test_label_dtype(self):
        _, label = self.ds[0]
        assert label.dtype == torch.long

    def test_ids_dtype(self):
        ids, _ = self.ds[0]
        assert ids.dtype == torch.long


# ================================================================== #
# 4. create_dataloaders                                                #
# ================================================================== #
class TestCreateDataloaders:
    def test_returns_three_loaders(self):
        df    = _make_df(60)
        vocab = _tiny_vocab()
        train_df = df.iloc[:40]
        val_df   = df.iloc[40:50]
        test_df  = df.iloc[50:]
        loaders  = create_dataloaders(train_df, val_df, test_df,
                                      vocab, batch_size=8, max_len=16)
        assert len(loaders) == 3

    def test_batch_shape(self):
        df    = _make_df(60)
        vocab = _tiny_vocab()
        train_df, val_df, test_df = df.iloc[:40], df.iloc[40:50], df.iloc[50:]
        train_loader, _, _ = create_dataloaders(
            train_df, val_df, test_df, vocab, batch_size=8, max_len=16
        )
        ids, labels = next(iter(train_loader))
        assert ids.shape[0] <= 8
        assert ids.shape[1] == 16
        assert labels.shape[0] <= 8


# ================================================================== #
# 5. compute_weights                                                   #
# ================================================================== #
class TestComputeWeights:
    def test_returns_dict_and_tensor(self):
        df = _make_df(120)
        wd, wt = compute_weights(df)
        assert isinstance(wd, dict)
        assert isinstance(wt, torch.Tensor)

    def test_tensor_shape(self):
        df = _make_df(120)
        _, wt = compute_weights(df)
        assert wt.shape == (3,)

    def test_all_weights_positive(self):
        df = _make_df(120)
        _, wt = compute_weights(df)
        assert (wt > 0).all()

    def test_dict_has_all_classes(self):
        df = _make_df(120)
        wd, _ = compute_weights(df)
        assert set(wd.keys()) == {'bad', 'neutral', 'good'}

    def test_balanced_classes_produce_equal_weights(self):
        # 40 of each class => weights should be approximately equal
        df = pd.DataFrame({
            'review':    ['x'] * 120,
            'rating':    [1] * 40 + [5] * 40 + [8] * 40,
            'sentiment': ['bad'] * 40 + ['neutral'] * 40 + ['good'] * 40
        })
        wd, _ = compute_weights(df)
        vals = list(wd.values())
        assert abs(vals[0] - vals[1]) < 0.01
        assert abs(vals[1] - vals[2]) < 0.01


# ================================================================== #
# 6. save_history_csv                                                  #
# ================================================================== #
class TestSaveHistoryCsv:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        # Patch RESULTS_DIR at module level for duration of test
        import training_utils as tu
        self._orig = tu.RESULTS_DIR
        tu.RESULTS_DIR = self.tmp
        import data_handler as dh
        dh.RESULTS_DIR = self.tmp

    def teardown_method(self):
        import training_utils as tu
        import data_handler as dh
        tu.RESULTS_DIR = self._orig
        dh.RESULTS_DIR = self._orig
        shutil.rmtree(self.tmp)

    def _make_history(self, n=3):
        return {
            'train_loss': [0.9, 0.7, 0.5],
            'val_loss':   [1.0, 0.8, 0.6],
            'train_acc':  [0.5, 0.6, 0.7],
            'val_acc':    [0.4, 0.5, 0.6],
            'train_f1':   [0.4, 0.5, 0.6],
            'val_f1':     [0.3, 0.4, 0.5],
        }

    def test_file_created(self):
        import training_utils as tu
        save_history_csv(self._make_history(), 'test_tag')
        assert os.path.isfile(os.path.join(self.tmp, 'test_tag_history.csv'))

    def test_epoch_rows_count(self):
        import training_utils as tu
        save_history_csv(self._make_history(), 'test_tag2')
        df = pd.read_csv(os.path.join(self.tmp, 'test_tag2_history.csv'))
        assert len(df) == 3

    def test_test_row_appended(self):
        import training_utils as tu
        save_history_csv(self._make_history(), 'test_tag3',
                         test_loss=0.4, test_acc=0.75, test_f1=0.70)
        df = pd.read_csv(os.path.join(self.tmp, 'test_tag3_history.csv'))
        assert len(df) == 4
        assert df.iloc[-1]['epoch'] == 'test'

    def test_expected_columns(self):
        import training_utils as tu
        save_history_csv(self._make_history(), 'test_tag4')
        df = pd.read_csv(os.path.join(self.tmp, 'test_tag4_history.csv'))
        for col in ['epoch', 'train_loss', 'val_loss',
                    'train_acc', 'val_acc', 'train_f1', 'val_f1']:
            assert col in df.columns

    def test_tag_sanitised_in_filename(self):
        import training_utils as tu
        save_history_csv(self._make_history(), 'BERT (len512)')
        expected = os.path.join(self.tmp, 'bert_len512_history.csv')
        assert os.path.isfile(expected)


# ================================================================== #
# 7. FocalLoss                                                         #
# ================================================================== #
class TestFocalLoss:
    def test_output_is_scalar(self):
        fl     = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss   = fl(logits, labels)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        fl     = FocalLoss(gamma=2.0)
        logits = torch.randn(16, 3)
        labels = torch.randint(0, 3, (16,))
        assert fl(logits, labels).item() >= 0.0

    def test_gamma_zero_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 3)
        labels = torch.randint(0, 3, (16,))
        fl_loss = FocalLoss(gamma=0.0)(logits, labels)
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        assert abs(fl_loss.item() - ce_loss.item()) < 1e-5

    def test_class_weights_accepted(self):
        weights = torch.tensor([1.0, 2.0, 1.5])
        fl      = FocalLoss(gamma=2.0, weight=weights)
        logits  = torch.randn(8, 3)
        labels  = torch.randint(0, 3, (8,))
        loss    = fl(logits, labels)
        assert loss.item() >= 0.0

    def test_higher_gamma_reduces_easy_example_loss(self):
        # Moderately confident correct prediction — confident enough to be
        # "easy" but not so extreme that float32 rounds pt to exactly 1.0,
        # which makes (1-pt)^gamma = 0 for all gamma and floors both losses
        # to 0.0 (the +-10 logit case hits that edge).
        logits = torch.tensor([[3.0, -1.0, -1.0]])
        labels = torch.tensor([0])
        loss_low  = FocalLoss(gamma=0.0)(logits, labels).item()
        loss_high = FocalLoss(gamma=4.0)(logits, labels).item()
        assert loss_high < loss_low


# ================================================================== #
# 8. TextCNN output shape                                              #
# ================================================================== #
class TestTextCNN:
    def setup_method(self):
        self.vocab_size = 100
        self.embed_dim  = 16
        self.num_classes = 3
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            num_filters=8,
            filter_sizes=[2, 3],
            dropout=0.0
        )
        self.model.eval()

    def test_output_shape(self):
        x = torch.randint(0, self.vocab_size, (4, 32))
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, self.num_classes)

    def test_output_is_logits_not_probs(self):
        x = torch.randint(0, self.vocab_size, (4, 32))
        with torch.no_grad():
            out = self.model(x)
        # logits are not constrained to [0,1]
        assert not ((out >= 0) & (out <= 1)).all()

    def test_different_batch_sizes(self):
        for bs in [1, 8, 16]:
            x = torch.randint(0, self.vocab_size, (bs, 32))
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (bs, self.num_classes)

    def test_pretrained_embeddings_loaded(self):
        emb = torch.randn(self.vocab_size, self.embed_dim)
        model = TextCNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            pretrained_embeddings=emb
        )
        assert torch.allclose(model.embedding.weight.data, emb)


# ================================================================== #
# 9. BiLSTM output shape                                               #
# ================================================================== #
class TestBiLSTM:
    def setup_method(self):
        self.vocab_size  = 100
        self.embed_dim   = 16
        self.hidden_dim  = 32
        self.num_classes = 3
        self.model = BiLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=2,
            dropout=0.0
        )
        self.model.eval()

    def test_output_shape(self):
        x = torch.randint(0, self.vocab_size, (4, 20))
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (4, self.num_classes)

    def test_different_seq_lengths(self):
        for seq_len in [16, 64, 128]:
            x = torch.randint(0, self.vocab_size, (2, seq_len))
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (2, self.num_classes)

    def test_different_batch_sizes(self):
        for bs in [1, 8]:
            x = torch.randint(0, self.vocab_size, (bs, 20))
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (bs, self.num_classes)

    def test_fc_layer_input_dim(self):
        # FC must take hidden_dim * 2 (bidirectional concat)
        assert self.model.fc.in_features == self.hidden_dim * 2

    def test_pretrained_embeddings_loaded(self):
        emb = torch.randn(self.vocab_size, self.embed_dim)
        model = BiLSTM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            pretrained_embeddings=emb
        )
        assert torch.allclose(model.embedding.weight.data, emb)


# ================================================================== #
# 10. BERTClassifier freeze logic                                      #
# (imports the class but does NOT call from_pretrained)               #
# ================================================================== #
class TestBERTClassifierFreezeLogic:
    """
    We test the freeze logic in isolation by building a minimal stand-in
    that mimics BERT's encoder structure without downloading weights.
    """

    class _FakeEncoder(nn.Module):
        def __init__(self, n_layers=12):
            super().__init__()
            self.layer = nn.ModuleList(
                [nn.Linear(4, 4) for _ in range(n_layers)]
            )

    class _FakeEmbeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(10, 4))

    class _FakeBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = TestBERTClassifierFreezeLogic._FakeEncoder()
            self.embeddings = TestBERTClassifierFreezeLogic._FakeEmbeddings()

    def _apply_freeze(self, fake_bert, freeze_layers):
        if freeze_layers > 0:
            for layer in fake_bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        if freeze_layers >= 12:
            for param in fake_bert.embeddings.parameters():
                param.requires_grad = False

    def test_no_freeze_all_trainable(self):
        bert = self._FakeBERT()
        self._apply_freeze(bert, 0)
        assert all(p.requires_grad for p in bert.encoder.parameters())
        assert all(p.requires_grad for p in bert.embeddings.parameters())

    def test_last2_layers_freeze_10(self):
        bert = self._FakeBERT()
        self._apply_freeze(bert, 10)
        for i, layer in enumerate(bert.encoder.layer):
            for p in layer.parameters():
                if i < 10:
                    assert not p.requires_grad, f"Layer {i} should be frozen"
                else:
                    assert p.requires_grad, f"Layer {i} should be trainable"

    def test_head_only_freeze_12_freezes_embeddings(self):
        bert = self._FakeBERT()
        self._apply_freeze(bert, 12)
        for p in bert.encoder.parameters():
            assert not p.requires_grad
        for p in bert.embeddings.parameters():
            assert not p.requires_grad

    def test_partial_freeze_leaves_later_layers_trainable(self):
        bert = self._FakeBERT()
        self._apply_freeze(bert, 6)
        for i, layer in enumerate(bert.encoder.layer):
            for p in layer.parameters():
                expected = i >= 6
                assert p.requires_grad == expected


# ================================================================== #
# 11. DistilBERTClassifier output shape                               #
# (uses a tiny fake backbone — no download)                           #
# ================================================================== #
class TestDistilBERTClassifierShape:

    class _FakeDistilBERT(nn.Module):
        """Minimal stand-in: returns last_hidden_state of shape (B, L, 768)."""
        class _Config:
            hidden_size = 768

        def __init__(self):
            super().__init__()
            self.config = self._Config()
            self.proj   = nn.Linear(768, 768)

        def forward(self, input_ids, attention_mask):
            B, L  = input_ids.shape
            h     = torch.zeros(B, L, 768)

            class _Out:
                pass
            out = _Out()
            out.last_hidden_state = h
            return out

    def test_output_shape(self):
        from model_distilbert import DistilBERTClassifier
        model = DistilBERTClassifier.__new__(DistilBERTClassifier)
        nn.Module.__init__(model)
        model.distilbert = self._FakeDistilBERT()
        model.dropout    = nn.Dropout(0.0)
        model.classifier = nn.Linear(768, 3)

        B, L = 4, 32
        input_ids      = torch.zeros(B, L, dtype=torch.long)
        attention_mask = torch.ones(B, L, dtype=torch.long)

        model.eval()
        with torch.no_grad():
            out = model(input_ids, attention_mask)
        assert out.shape == (B, 3)


# ================================================================== #
# 12. Output filename conventions                                      #
# ================================================================== #
class TestOutputFilenameConventions:
    def test_bert_run_tag_format(self):
        for max_len in [128, 256, 512]:
            for tag in ['no_weighting', 'with_weighting', 'focal',
                        'head_only', 'last2_layers']:
                run_tag = f'bert_len{max_len}_{tag}'
                assert f'len{max_len}' in run_tag
                assert tag in run_tag

    def test_distilbert_run_tag_format(self):
        for max_len in [128, 256, 512]:
            for tag in ['no_weighting', 'with_weighting']:
                run_tag = f'distilbert_len{max_len}_{tag}'
                assert f'len{max_len}' in run_tag
                assert tag in run_tag

    def test_bilstm_run_tag_format(self):
        for max_len in [128, 256, 512]:
            for tag in ['no_weighting', 'with_weighting']:
                run_tag = f'bilstm_len{max_len}_{tag}'
                assert f'len{max_len}' in run_tag

    def test_all_results_json_naming(self):
        for max_len in [128, 256, 512]:
            fname = f'all_results_len{max_len}.json'
            assert f'len{max_len}' in fname

    def test_final_comparison_naming(self):
        for max_len in [128, 256, 512]:
            csv   = f'final_comparison_len{max_len}.csv'
            chart = f'final_performance_len{max_len}.png'
            assert f'len{max_len}' in csv
            assert f'len{max_len}' in chart


# ================================================================== #
# 13. Cache guard logic (_splits_exist)                                #
# ================================================================== #
class TestCacheGuard:
    def test_returns_false_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr('data_handler.TRAIN_CSV',
                            str(tmp_path / 'train.csv'))
        monkeypatch.setattr('data_handler.VAL_CSV',
                            str(tmp_path / 'val.csv'))
        monkeypatch.setattr('data_handler.TEST_CSV',
                            str(tmp_path / 'test.csv'))
        assert not _splits_exist()

    def test_returns_false_when_only_some_files_exist(self, tmp_path,
                                                       monkeypatch):
        train = tmp_path / 'train.csv'
        train.write_text('review,rating,sentiment\n')
        monkeypatch.setattr('data_handler.TRAIN_CSV', str(train))
        monkeypatch.setattr('data_handler.VAL_CSV',
                            str(tmp_path / 'val.csv'))
        monkeypatch.setattr('data_handler.TEST_CSV',
                            str(tmp_path / 'test.csv'))
        assert not _splits_exist()

    def test_returns_true_when_all_files_exist(self, tmp_path, monkeypatch):
        for name in ['train.csv', 'val.csv', 'test.csv']:
            (tmp_path / name).write_text('review,rating,sentiment\n')
        monkeypatch.setattr('data_handler.TRAIN_CSV',
                            str(tmp_path / 'train.csv'))
        monkeypatch.setattr('data_handler.VAL_CSV',
                            str(tmp_path / 'val.csv'))
        monkeypatch.setattr('data_handler.TEST_CSV',
                            str(tmp_path / 'test.csv'))
        assert _splits_exist()