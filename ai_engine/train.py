import os
import sys
import json
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ai_engine.model import IntentClassifier, Tokenizer
from ai_engine.dataset import EcoDataset, collate_fn

CONFIG = {
    "data_path": os.path.join(PROJECT_ROOT, "ai_engine", "data", "eco_dataset.json"),
    "save_dir": os.path.join(PROJECT_ROOT, "ai_engine", "checkpoints"),
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "max_len": 64,
    "epochs": 100,
    "batch_size": 16,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "seed": 42,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_intent_classifier():
    print("=" * 60)
    print("🌿 EcoMind — Обучение модели")
    print("=" * 60)

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Устройство: {device}")

    print("\n📦 Загрузка датасета...")
    tokenizer = Tokenizer()
    dataset = EcoDataset(
        CONFIG["data_path"], tokenizer, max_len=CONFIG["max_len"], augment=True
    )
    print(f"   Классов: {dataset.num_classes}")
    print(f"   Сэмплов: {len(dataset)}")
    print(f"   Словарь: {tokenizer.vocab_size} токенов")
    print(f"   Теги: {list(dataset.tag2idx.keys())}")

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 2. Создание модели
    model = IntentClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=dataset.num_classes,
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_layers=CONFIG["num_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_len"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Модель: {total_params:,} параметров ({trainable_params:,} обучаемых)")

    # 3. Обучение
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Начинаем обучение ({CONFIG['epochs']} эпох)...\n")

    best_acc = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_mask, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        acc = correct / total
        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"   Эпоха {epoch+1:3d}/{CONFIG['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {acc:.1%} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        if acc > best_acc:
            best_acc = acc

    print(f"\n✅ Лучшая accuracy: {best_acc:.1%}")
    return model, tokenizer, dataset

def build_tfidf_index(dataset: EcoDataset):
    print("\n📊 Построение TF-IDF индекса...")

    with open(CONFIG["data_path"], "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    patterns = []
    pattern_tags = []
    for item in raw_data:
        for p in item["patterns"]:
            patterns.append(p.lower())
            pattern_tags.append(item["tag"])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(patterns)

    print(f"   Паттернов: {len(patterns)}")
    print(f"   Признаков: {tfidf_matrix.shape[1]}")

    return vectorizer, tfidf_matrix, patterns, pattern_tags

def save_all(model, tokenizer, dataset, vectorizer, tfidf_matrix, patterns, pattern_tags):
    save_dir = CONFIG["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "intent_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "num_classes": dataset.num_classes,
                "d_model": CONFIG["d_model"],
                "nhead": CONFIG["nhead"],
                "num_layers": CONFIG["num_layers"],
                "dim_feedforward": CONFIG["dim_feedforward"],
                "dropout": CONFIG["dropout"],
                "max_len": CONFIG["max_len"],
            },
        },
        model_path,
    )
    print(f"   💾 Модель: {model_path}")

    tok_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"   💾 Токенизатор: {tok_path}")

    mappings_path = os.path.join(save_dir, "mappings.json")
    with open(mappings_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag2idx": dataset.tag2idx,
                "idx2tag": {str(k): v for k, v in dataset.idx2tag.items()},
                "tag_responses": dataset.tag_responses,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"   💾 Маппинги: {mappings_path}")

    tfidf_path = os.path.join(save_dir, "tfidf.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "matrix": tfidf_matrix,
                "patterns": patterns,
                "pattern_tags": pattern_tags,
            },
            f,
        )
    print(f"   💾 TF-IDF: {tfidf_path}")

def test_model(model, tokenizer, dataset, vectorizer, tfidf_matrix, patterns, pattern_tags):
    from ai_engine.inference import EcoMindEngine

    print("\n" + "=" * 60)
    print("🧪 Тестирование модели")
    print("=" * 60)

    engine = EcoMindEngine(CONFIG["save_dir"])

    test_queries = [
        "как рассчитать углеродный след",
        "какой транспорт самый экологичный",
        "расскажи про электромобили",
        "что такое net zero",
        "как заводы загрязняют природу",
        "сколько CO2 выбрасывает самолёт",
        "привет",
        "как озеленить город",
        "что есть чтобы меньше вредить экологии",
    ]

    for q in test_queries:
        result = engine.answer(q)
        print(f"\n❓ {q}")
        print(f"   🏷️  Интент: {result['intent']} (confidence: {result['confidence']:.1%})")
        print(f"   📝 Ответ: {result['response'][:100]}...")

if __name__ == "__main__":
    model, tokenizer, dataset = train_intent_classifier()
    vectorizer, tfidf_matrix, patterns, pattern_tags = build_tfidf_index(dataset)
    print("\n💾 Сохранение модели...")
    save_all(model, tokenizer, dataset, vectorizer, tfidf_matrix, patterns, pattern_tags)
    test_model(model, tokenizer, dataset, vectorizer, tfidf_matrix, patterns, pattern_tags)
    print("\n" + "=" * 60)
    print("✅ Обучение завершено! Модель готова к использованию.")
    print("=" * 60)
