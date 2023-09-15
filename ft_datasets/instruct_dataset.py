# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "test_input": (
        # TODO: Do I need to change "prompt_input?"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=30):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        # TODO: Find nicer solution than just removing words from the input to assert tokens is not > max_tokens
        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        while len(example) > self.max_tokens:
            words = ann["input"].split()
            ann["input"] = " ".join(words[:-1])
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]
            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            example = self.tokenizer.encode(example)
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
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


class TestingDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=30):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
       

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        # TODO: Dig deeper into the dataset code
        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        while len(prompt) > self.max_tokens:
            words = ann["input"].split()
            ann["input"] = " ".join(words[:-1])
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            prompt = self.tokenizer.encode(prompt)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_tokens - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # batch = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_tokens,
        #     return_tensors="pt",
        # )

        # while len(batch["input_ids"]) > self.max_tokens:
        #     ann = self.ann[index]
        #     words = ann["input"].split()
        #     ann["input"] = " ".join(words[:-1])
        #     prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        #     batch = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_tokens,
        #     return_tensors="pt",
        # )

        # return batch
        return {
            "input_ids": prompt
        }
