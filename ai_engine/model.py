"""
EcoMind AI Model — Гибридная архитектура:
  1. Intent Classifier (нейросеть) — определяет тему вопроса
  2. TF-IDF Retrieval — семантический поиск ближайшего вопроса
  3. Response Generator — выбирает лучший ответ

Это production-ready модель для хакатона.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class IntentClassifier(nn.Module):
    """
    Нейросетевой классификатор интентов (тем) пользовательских вопросов.
    Архитектура: Embedding → Transformer Encoder → Classification Head
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) — индексы токенов
            mask: (batch, seq_len) — маска паддинга (True = игнорировать)
        Returns:
            logits: (batch, num_classes)
        """
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)

        if mask is not None:
            output = self.transformer_encoder(emb, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(emb)

        # Global average pooling (игнорируя паддинг)
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()  # (batch, seq, 1)
            output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

        logits = self.classifier(output)
        return logits


class Tokenizer:
    """
    Простой символьно-словный токенизатор для русского текста.
    Для хакатона этого достаточно. В продакшене используйте SentencePiece/BPE.
    """

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.next_idx = 2

    def fit(self, texts: list[str]):
        """Построить словарь из текстов"""
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.word2idx:
                    self.word2idx[word] = self.next_idx
                    self.idx2word[self.next_idx] = word
                    self.next_idx += 1

    def encode(self, text: str, max_len: int = 128) -> list[int]:
        """Текст → список индексов"""
        tokens = self._tokenize(text)
        indices = [
            self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens
        ]
        # Обрезка / паддинг
        if len(indices) > max_len:
            indices = indices[:max_len]
        return indices

    def _tokenize(self, text: str) -> list[str]:
        """Простая токенизация: lowercase + split по пробелам и пунктуации"""
        text = text.lower().strip()
        # Заменяем пунктуацию пробелами
        for ch in ".,!?;:()[]{}\"'—–-/\\@#$%^&*+=<>~`":
            text = text.replace(ch, " ")
        return [w for w in text.split() if w]

    @property
    def vocab_size(self) -> int:
        return self.next_idx

    def save(self, path: str):
        """Сохранить словарь"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)

    def load(self, path: str):
        """Загрузить словарь"""
        import json
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.next_idx = max(self.word2idx.values()) + 1
