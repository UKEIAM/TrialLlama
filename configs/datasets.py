# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class ct_v2:
    dataset: str = "ct_v2"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_v2.json"


@dataclass
class ct_v3:
    dataset: str = "ct_v3"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_v3.json"


# Test data, never seen by the model
@dataclass
class ct_testing_v2:
    dataset: str = "ct_testing_v2"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_testing_v2.json"


@dataclass
class ct_testing_v3:
    dataset: str = "ct_testing_v3"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_testing_v3.json"


@dataclass
class ct_reasoning:
    dataset: str = "ct_reasoning"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_reasoning_full_v2.json"
