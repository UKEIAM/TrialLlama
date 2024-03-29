# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


PROMPT_DICT = {
    "prompt_input": ("{instruction}\n\n{topic}\n\n{clinical_trial}\n\n{response}\n\n")
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=1024):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann, _ = train_test_split(self.ann, test_size=0.1, random_state=42)
        else:
            _, self.ann = train_test_split(self.ann, test_size=0.1, random_state=42)

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        # example_mask = example_mask.float()

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


class TestingDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="test", max_tokens=30):
        self.ann = json.load(open(dataset_config.data_path))
        self.ann = self.ann

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # NOTE: Funny, the approach used for tokenization in the training phase, is not working if you call model.generate(**batch). CUDA fails somehow
        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        batch = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )

        return batch
