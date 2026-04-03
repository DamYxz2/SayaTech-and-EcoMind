"""
EcoMind Inference Engine — гибридный AI:
  1. Своя нейросеть (Transformer) определяет тему
  2. TF-IDF находит ближайший вопрос из базы знаний
  3. Если уверенность низкая → ищет в интернете (Wikipedia, DuckDuckGo, Google)

Полностью автономный — без API ключей.
"""

import json
import os
import pickle
import random
import logging

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ai_engine.model import IntentClassifier, Tokenizer

logger = logging.getLogger(__name__)

# Порог уверенности — если ниже, идём в интернет
CONFIDENCE_THRESHOLD = 0.45


class EcoMindEngine:
    """
    Главный движок чат-бота.
    """

    def __init__(self, checkpoint_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(checkpoint_dir)
        self._load_tfidf(checkpoint_dir)
        self._load_mappings(checkpoint_dir)

        # Веб-поиск загружаем лениво
        self._web_search = None

    def _get_web_search(self):
        """Ленивая загрузка модуля веб-поиска"""
        if self._web_search is None:
            try:
                from ai_engine.web_search import web_search
                self._web_search = web_search
                logger.info("Веб-поиск подключён")
            except ImportError as e:
                logger.warning(
                    f"Веб-поиск недоступен (установите requests и beautifulsoup4): {e}"
                )
                self._web_search = False  # Пометить как недоступный
        return self._web_search if self._web_search else None

    def _load_model(self, checkpoint_dir: str):
        """Загрузка нейросети"""
        self.tokenizer = Tokenizer()
        self.tokenizer.load(os.path.join(checkpoint_dir, "tokenizer.json"))

        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "intent_model.pt"),
            map_location=self.device,
            weights_only=False,
        )
        cfg = checkpoint["config"]
        self.model = IntentClassifier(
            vocab_size=cfg["vocab_size"],
            num_classes=cfg["num_classes"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            max_len=cfg["max_len"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.max_len = cfg["max_len"]

    def _load_tfidf(self, checkpoint_dir: str):
        """Загрузка TF-IDF индекса"""
        with open(os.path.join(checkpoint_dir, "tfidf.pkl"), "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        self.tfidf_patterns = data["patterns"]
        self.tfidf_tags = data["pattern_tags"]

    def _load_mappings(self, checkpoint_dir: str):
        """Загрузка маппингов"""
        with open(
            os.path.join(checkpoint_dir, "mappings.json"), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        self.tag2idx = data["tag2idx"]
        self.idx2tag = {int(k): v for k, v in data["idx2tag"].items()}
        self.tag_responses = data["tag_responses"]

    def _predict_intent_nn(self, text: str) -> tuple:
        """Предсказание нейросетью"""
        encoded = self.tokenizer.encode(text, self.max_len)
        x = torch.tensor([encoded], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_idx = probs.max(dim=-1)

        tag = self.idx2tag[pred_idx.item()]
        return tag, confidence.item()

    def _predict_intent_tfidf(self, text: str) -> tuple:
        """Предсказание через TF-IDF"""
        query_vec = self.vectorizer.transform([text.lower()])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        tag = self.tfidf_tags[best_idx]
        return tag, float(best_score)

    def _search_web(self, query: str) -> dict:
        """Поиск в интернете"""
        search_fn = self._get_web_search()
        if search_fn:
            try:
                result = search_fn(query)
                if result["found"]:
                    return {
                        "intent": "web_search",
                        "confidence": 0.8,
                        "response": result["response"],
                        "method": f"web:{result['source']}",
                    }
            except Exception as e:
                logger.warning(f"Web search failed: {e}")

        return None

    def answer(self, question: str) -> dict:
        """
        Главный метод — генерация ответа.

        Логика:
        1. Нейросеть + TF-IDF пытаются классифицировать вопрос
        2. Если уверенность высокая → ответ из базы знаний
        3. Если уверенность низкая → поиск в интернете
        4. Если интернет не помог → ответ из базы (лучшее что есть)
        """
        # ── Шаг 1: Классификация ──
        nn_tag, nn_conf = self._predict_intent_nn(question)
        tfidf_tag, tfidf_conf = self._predict_intent_tfidf(question)

        # Ансамбль
        if nn_tag == tfidf_tag:
            final_tag = nn_tag
            final_conf = max(nn_conf, tfidf_conf)
            method = "ensemble"
        elif nn_conf > 0.7:
            final_tag = nn_tag
            final_conf = nn_conf
            method = "neural_net"
        elif tfidf_conf > 0.3:
            final_tag = tfidf_tag
            final_conf = tfidf_conf
            method = "tfidf"
        else:
            final_tag = nn_tag
            final_conf = nn_conf * 0.5
            method = "low_conf"

        # ── Шаг 2: Решение — база знаний или интернет ──

        # Если уверенность высокая и тема не "unknown" → база знаний
        if final_conf >= CONFIDENCE_THRESHOLD and final_tag != "unknown":
            responses = self.tag_responses.get(final_tag, [])
            if responses:
                return {
                    "intent": final_tag,
                    "confidence": final_conf,
                    "response": random.choice(responses),
                    "method": method,
                }

        # ── Шаг 3: Поиск в интернете ──
        web_result = self._search_web(question)
        if web_result:
            return web_result

        # ── Шаг 4: Fallback — лучший ответ из базы ──
        responses = self.tag_responses.get(final_tag, [])
        if responses:
            return {
                "intent": final_tag,
                "confidence": final_conf,
                "response": random.choice(responses),
                "method": f"{method}:fallback",
            }

        return {
            "intent": "unknown",
            "confidence": 0.0,
            "response": (
                "Я пока не знаю ответа на этот вопрос. "
                "Попробуйте переформулировать или спросите о другой теме: "
                "экология, математика, физика, история, программирование, здоровье."
            ),
            "method": "no_answer",
        }
