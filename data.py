import gc
import json
from dataclasses import dataclass, field
from typing import List, T_co, Dict
from collections import Counter
import pytorch_lightning as plt
import torch
import logging
import math
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import T5Tokenizer


@dataclass
class Example:
    question: str = field(default=None)
    answer: str = field(default=None)


class REDataset(Dataset):
    def __init__(self, max_token: int, model_name: str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.examples: List[Example] = []
        self.max_token = max_token

    def init_data(self, input_data: List[Example]):
        self.examples = input_data

    def __getitem__(self, index) -> T_co:
        question = self.tokenizer(self.examples[index].question).input_ids
        if len(question) > self.max_token:
            question = question[:self.max_token - 1] + [self.tokenizer.eos_token_id]
        if len(self.examples[index].answer) > self.max_token:
            logging.info(f'Long target detected: {len(self.examples[index].answer)}')
        return {
            'question': question,
            'answer': self.tokenizer(self.examples[index].answer).input_ids,
            'pad_token_id': self.tokenizer.pad_token_id
        }

    def remove(self, ex: Example):
        self.examples.remove(ex)

    def __len__(self):
        return len(self.examples)


class REDataModule(plt.LightningDataModule):

    def __init__(self,
                 model_name: str,
                 train_path: str,
                 valid_path: str,
                 batch_size: int,
                 max_token: int,
                 num_workers: int,
                 weighted: bool,
                 alpha: float):
        super(REDataModule, self).__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.train_data = REDataset(max_token=max_token, model_name=model_name)
        self.valid_data = REDataset(max_token=max_token, model_name=model_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted = weighted
        self.alpha = alpha
        self.train_weights = []

    def setup(self, stage=None) -> None:
        if self.weighted:
            train_class = []
            with open(self.train_path) as f:
                for line in f:
                    jsonline = json.loads(line)
                    train_class.append(jsonline['answer'].strip())
            class_counter = Counter(train_class)
            class_weights = {}
            for key, item in class_counter.items():
                class_weights[key] = math.pow(item / len(train_class), self.alpha)
            normalize_den = sum([item for _, item in class_weights.items()])
            for key, item in class_counter.items():
                class_weights[key] = (class_weights[key] / normalize_den) / class_counter[key]
            for ex in train_class:
                self.train_weights.append(class_weights[ex])
        train_ex = []
        valid_ex = []
        with open(self.train_path) as f:
            for line in f:
                jsonline = json.loads(line)
                ex = Example(question=jsonline['question'], answer=jsonline['answer'])
                train_ex.append(ex)
        self.train_data.init_data(train_ex)
        with open(self.valid_path) as f:
            for line in f:
                jsonline = json.loads(line)
                ex = Example(question=jsonline['question'], answer=jsonline['answer'])
                valid_ex.append(ex)
        self.valid_data.init_data(valid_ex)

    def collate_fn(self, batch: List) -> Dict:
        pad_token_id = batch[0]['pad_token_id']
        src_max_seq = max([len(ex['question']) for ex in batch])
        src_tensor = torch.full((len(batch), src_max_seq), pad_token_id)
        attention_mask = torch.zeros(len(batch), src_max_seq, dtype=torch.long)

        tgt_max_seq = max([len(ex['answer']) for ex in batch])
        tgt_tensor = torch.zeros(len(batch), tgt_max_seq, dtype=torch.long)
        tgt_attention_mask = torch.zeros(len(batch), tgt_max_seq, dtype=torch.long)
        for idx, ex in enumerate(batch):
            src_tensor[idx, :len(ex['question'])] = torch.tensor(ex['question'], dtype=torch.long)
            attention_mask[idx, :len(ex['question'])] = torch.ones(len(ex['question']))
            tgt_tensor[idx, :len(ex['answer'])] = torch.tensor(ex['answer'], dtype=torch.long)
            tgt_attention_mask[idx, :len(ex['answer'])] = torch.ones(len(ex['answer']))

        return {
            'src_tensor': src_tensor,
            'attention_mask': attention_mask,
            'tgt_tensor': tgt_tensor,
            'tgt_attention_mask': tgt_attention_mask,
        }

    def train_dataloader(self) -> DataLoader:
        if self.weighted:
            weighted_random_sampler = WeightedRandomSampler(
                weights=self.train_weights, num_samples=len(self.train_data))
            return DataLoader(self.train_data, sampler=weighted_random_sampler, pin_memory=True,
                              num_workers=self.num_workers, collate_fn=self.collate_fn,
                              batch_size=self.batch_size)
        return DataLoader(self.train_data, shuffle=True, pin_memory=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, shuffle=False, pin_memory=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, batch_size=self.batch_size)
