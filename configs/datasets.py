# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class ct:
    dataset: str = "ct"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct.json"


# Test data, never seen by the model
@dataclass
class ct_testing:
    dataset: str = "ct_testing"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_testing.json"
