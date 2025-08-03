import json
import pickle


def get_data(json_path) -> tuple:
    texts = []
    metadatas = []

    with open(json_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    for article in articles:
        for key, content in article.items():
            if key == "metadata":
                current_meta = content
                metadatas.append(current_meta)


            else:
                text = f"{key}\n{content}"
                texts.append(text)

    return texts, metadatas


import json
import spacy
nlp = spacy.load("fr_core_news_sm")

def sent_tokenize_spacy(text):
    doc = nlp(text)
    filtered_sents = []

    for sent in doc.sents:
        cleaned = sent.text.strip()

        # Filter out short or irrelevant tokens like "Art.", "2", "3"
        if len(cleaned) < 20:  # too short to be meaningful
            continue
        elif cleaned.lower().startswith("art."):
            continue
        elif cleaned.isdigit():
            continue

        else:
            filtered_sents.append(cleaned)

    return filtered_sents

import os
import pickle
import random
from tqdm import tqdm

def prepare_nsp_data(texts, load_data=True, save_data=False, ratio_negative=0.5):
    sentence_pairs = []

    data_dir = "nsp_cache"
    os.makedirs(data_dir, exist_ok=True)

    all_sentences_path = os.path.join(data_dir, "all_sentences.pkl")
    all_texts_tokenized_path = os.path.join(data_dir, "all_texts_tokenized.pkl")

    if load_data and os.path.exists(all_sentences_path) and os.path.exists(all_texts_tokenized_path):
        print("Loading tokenized data from disk...")
        with open(all_sentences_path, "rb") as f:
            all_sentences = pickle.load(f)
        with open(all_texts_tokenized_path, "rb") as f:
            all_texts_tokenized = pickle.load(f)
    else:
        print("Tokenizing and preparing data...")
        all_sentences = []
        all_texts_tokenized = []

        for text in tqdm(texts):
            sents = sent_tokenize_spacy(text)
            all_sentences.extend(sents)
            all_texts_tokenized.append(sents)

        if save_data:
            print("Saving tokenized data to disk...")
            with open(all_sentences_path, "wb") as f:
                pickle.dump(all_sentences, f)
            with open(all_texts_tokenized_path, "wb") as f:
                pickle.dump(all_texts_tokenized, f)

    # Create positive and negative sentence pairs
    for sents in all_texts_tokenized:
        for i in range(len(sents) - 1):
            if random.random() > ratio_negative:
                # print(f'sent i {sents[i]}')
                # print(f'sent i + 1 {sents[i+1]}')
                sentence_pairs.append({
                    "sentence1": sents[i],
                    "sentence2": sents[i + 1],
                    "label": 1
                })
            else:
                rand_sent = random.choice(all_sentences)
#                 print(f'neg sent i {sents[i]}')
#                 print(f'neg sent i + 1 {rand_sent}')
                sentence_pairs.append({
                    "sentence1": sents[i],
                    "sentence2": rand_sent,
                    "label": 0
                })

    return sentence_pairs

from torch.utils.data import Dataset
import torch
class NSPDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, max_length=128):
        self.examples = []
        self.tokenizer = tokenizer

        for pair in sentence_pairs:
            encoding = tokenizer(
                pair["sentence1"],
                pair["sentence2"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            encoding["next_sentence_label"] = torch.tensor(pair["label"])
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.examples[idx].items()}
        return item
