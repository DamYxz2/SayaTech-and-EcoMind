import json
import os
import random
import torch
from torch.utils.data import Dataset


class EcoDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 128, augment: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.tag2idx = {}
        self.idx2tag = {}
        self.tag_responses = {}
        self.samples = []

        for item in raw_data:
            tag = item["tag"]
            if tag not in self.tag2idx:
                idx = len(self.tag2idx)
                self.tag2idx[tag] = idx
                self.idx2tag[idx] = tag

            self.tag_responses[tag] = item["responses"]

            for pattern in item["patterns"]:
                self.samples.append((pattern, self.tag2idx[tag]))

        all_texts = [s[0] for s in self.samples]
        for resp_list in self.tag_responses.values():
            all_texts.extend(resp_list)
        self.tokenizer.fit(all_texts)

        if augment:
            augmented = []
            for text, label in self.samples:
                augmented.append((text, label))
                words = text.split()
                if len(words) > 2:
                    drop_idx = random.randint(0, len(words) - 1)
                    new_text = " ".join(w for i, w in enumerate(words) if i != drop_idx)
                    augmented.append((new_text, label))
                if len(words) > 2:
                    shuffled = words.copy()
                    random.shuffle(shuffled)
                    augmented.append((" ".join(shuffled), label))
            self.samples = augmented

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoded = self.tokenizer.encode(text, self.max_len)
        return encoded, label

    @property
    def num_classes(self):
        return len(self.tag2idx)

    def get_response(self, tag: str) -> str:
        responses = self.tag_responses.get(tag, ["Извините, не могу ответить на этот вопрос."])
        return random.choice(responses)


def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)

    padded = []
    masks = []
    for t in texts:
        pad_len = max_len - len(t)
        padded.append(t + [0] * pad_len)
        masks.append([False] * len(t) + [True] * pad_len)

    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
    )
