import os
import random
import copy
from dataclasses import dataclass
from typing import List, Dict

import ujson as json
import torch
from torch.utils.data import Dataset, IterableDataset
import transformers

from data_utils.indexed_dataset import make_dataset as make_indexed_dataset

IGNORE_INDEX = -100


class LMIndexedDataset(Dataset):
    def __init__(self, indexed_dataset_prefix, tokenizer):
        super().__init__()
        self.indexed_dataset = make_indexed_dataset(indexed_dataset_prefix, "mmap")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, index):
        input_ids = self.indexed_dataset[index].astype(int)
        input_ids = input_ids[:self.tokenizer.model_max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = copy.deepcopy(input_ids)

        return dict(input_ids=input_ids, labels=labels)


class LMDataset(Dataset):
    def __init__(
        self,
        jsonl_file_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        """
        :param jsonl_file_path:
        :param tokenizer:
        :param prepend_bos: (default: True) Set as False when using GPT2.
        :param append_eos:  (default: True)
        """
        super().__init__()
        self.jsonl_file_path = jsonl_file_path
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.texts = []
        with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = json.loads(line)['text']
                self.texts.append(text)

    def preprocess(self, text, strategy='sliding_window'):
        tokens = self.tokenizer.tokenize(text)
        max_length = self.tokenizer.model_max_length - int(self.prepend_bos) - int(self.append_eos)

        if strategy == 'sliding_window':
            if len(tokens) > max_length:
                start_index = random.randint(0, len(tokens) - max_length)
                tokens = tokens[start_index: start_index + max_length]

        elif strategy == 'truncate':
            tokens = tokens[:max_length]

        else:
            raise ValueError(f'Unknown strategy: [{strategy}]')

        # special tokens
        if self.prepend_bos:
            tokens = [self.tokenizer.bos_token] + tokens
        if self.append_eos:
            tokens = tokens + [self.tokenizer.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # only for checking oom issue
        # input_ids += [self.tokenizer.pad_token_id] * (max_length - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = copy.deepcopy(input_ids)

        assert (len(input_ids) <= self.tokenizer.model_max_length)
        # print(self.tokenizer.vocab_size, len(self.tokenizer.vocab))   # TODO: Bug report - They are different
        # assert (max(input_ids) < len(self.tokenizer.vocab)), input_ids

        return dict(input_ids=input_ids, labels=labels)

    def __getitem__(self, index):
        text = self.texts[index]
        return self.preprocess(text)

    def __len__(self):
        return len(self.texts)


@dataclass
class LMDataCollator:
    tokenizer: transformers.PreTrainedTokenizer
    is_training: bool = True

    def __call__(self, batch):
        input_ids, labels = tuple([b[key] for b in batch] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # print(torch.max(input_ids))
        # print(torch.min(input_ids))
        # print(input_ids.shape)
        #
        # print('=====')
        # print(input_ids.numpy().tolist())
        # print(input_ids.ne(self.tokenizer.pad_token_id))
        # print('=====')
        #
        # exit(1)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class TaskDataset(Dataset):
    def __init__(
            self,
            samples: List[Dict],
            tokenizer: transformers.PreTrainedTokenizer,
            truncation_strategy=None,  # - / left / right / sliding_window
            objective="language_modeling"   # language_modeling / seq2seq
    ):
        """
        :param samples:
            each sample must include
                question
                options
                answer_idx

        :param tokenizer:

        :param truncation_strategy:
            None: do nothing, and wait for padding
            concatenation: concatenate samples until reaching the max_length
            left: truncate from the left side
            right: truncate from the right side
            sliding_window: truncate with sliding windows

        :param objective:
            language_modeling: do standard LM
            seq2seq: only predict the target text
        """

        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.all_input_ids, self.all_labels = self.prepare_all_data(
            truncation_strategy=truncation_strategy,
            objective=objective
        )

    def prepare_all_data(
        self,
        truncation_strategy=None,  # - / concatenation / left / right / sliding_window
        objective="language_modeling"   # language_modeling / seq2seq
    ):

        all_input_ids, all_labels = [], []
        cache = {"input_ids": [], "labels": []}

        for sample in self.samples:
            question = sample["question"]
            answer = sample["options"][sample["answer_idx"]]

            if objective == "language_modeling":
                text = f"Question: {question}\nAnswer: {answer}"
                encodings = self.tokenizer(text)
                input_ids = encodings["input_ids"] #  + [self.tokenizer.eos_token_id]
                labels = copy.deepcopy(input_ids)

                if truncation_strategy is None:   # do nothing
                    all_input_ids.append(input_ids)
                    all_labels.append(labels)

                elif truncation_strategy == "concatenation":   # concatenation
                    cache["input_ids"].extend(input_ids)
                    cache["labels"].extend(labels)

                    if len(cache["input_ids"]) >= self.tokenizer.model_max_length:
                        all_input_ids.append(cache["input_ids"])
                        all_labels.append(cache["labels"])
                        cache = {"input_ids": [], "labels": []}

                else:
                    raise ValueError(f'Not ready truncation_strategy: [{truncation_strategy}]')

            elif objective == "seq2seq":
                input_text = f"Question: {question}\nAnswer:"
                output_text = f" {answer}"

                input_length = len(self.tokenizer(input_text, return_length=True)) # + 1
                input_ids = self.tokenizer(input_text + output_text)["input_ids"] # + [self.tokenizer.eos_token_id]
                labels = copy.deepcopy(input_ids)
                labels[:input_length] = [-100] * input_length

                if truncation_strategy in [None, "left", "right"]:  # do nothing
                    all_input_ids.append(input_ids)
                    all_labels.append(labels)

                elif truncation_strategy == "concatenation":   # concatenation
                    cache["input_ids"].extend(input_ids)
                    cache["labels"].extend(labels)

                    if len(cache["input_ids"]) >= self.tokenizer.model_max_length:
                        all_input_ids.append(cache["input_ids"])
                        all_labels.append(cache["labels"])
                        cache = {"input_ids": [], "labels": []}

                else:
                    raise ValueError(f'Not ready truncation_strategy: [{truncation_strategy}]')

        if objective == "language_modeling":
            if len(cache["input_ids"]) > 0:
                all_input_ids.append(cache["input_ids"])
                all_labels.append(cache["labels"])
            cache = {"input_ids": [], "labels": []}

        elif objective == "seq2seq":
            if len(cache["input_ids"]) > 0:
                all_input_ids.append(cache["input_ids"])
                all_labels.append(cache["labels"])
            cache = {"input_ids": [], "labels": []}

        return all_input_ids, all_labels

    def preprocess(
        self,
        input_text,
        output_text=None,
        strategy='sliding_window',  # truncate / sliding_window
        truncation_side='right',  # left / right
        objective="language_modeling"   # language_modeling / seq2seq
    ):
        input_tokens = self.tokenizer.tokenize(input_text)
        max_length = self.tokenizer.model_max_length - int(self.prepend_bos) - int(self.append_eos)

        if strategy == 'sliding_window':
            if len(input_tokens) > max_length:
                start_index = random.randint(0, len(input_tokens) - max_length)
                tokens = input_tokens[start_index: start_index + max_length]

        elif strategy == 'truncate':
            if truncation_side == 'left':
                tokens = input_tokens[-max_length:]
            else:
                tokens = input_tokens[:max_length]

        else:
            raise ValueError(f'Unknown strategy: [{strategy}]')

        # special tokens
        if self.prepend_bos:
            tokens = [self.tokenizer.bos_token] + tokens
        if self.append_eos:
            tokens = tokens + [self.tokenizer.eos_token]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # only for checking oom issue
        # input_ids += [self.tokenizer.pad_token_id] * (max_length - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = copy.deepcopy(input_ids)

        assert (len(input_ids) <= self.tokenizer.model_max_length)
        # print(self.tokenizer.vocab_size, len(self.tokenizer.vocab))   # TODO: Bug report - They are different
        assert (max(input_ids) < len(self.tokenizer.vocab)), input_ids

        return dict(input_ids=input_ids, labels=labels)

    def __getitem__(self, index):
        input_ids = self.all_input_ids[index]
        labels = self.all_labels[index]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.all_input_ids)



















