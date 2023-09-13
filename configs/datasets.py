# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class clinical_trials_dataset:
    dataset: str = "clinical_trials_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/clinical_trials.json"


@dataclass
class ct_300:
    dataset: str = "ct_300"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_300.json"


@dataclass
class ct_10:
    dataset: str = "ct_10"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_10.json"


# Test data, never seen by the model
@dataclass
class clinical_trials_testing:
    dataset: str = "clinical_trials_testing"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/clinical_trials_testing.json"

# Test data, never seen by the model
@dataclass
class ct_testing_300:
    dataset: str = "ct_testing_300"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_testing_300.json"

# Test data, never seen by the model
@dataclass
class ct_testing_10:
    dataset: str = "ct_testing_10"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_testing_10.json"
