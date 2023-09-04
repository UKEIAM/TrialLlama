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
class ct_testing:
    dataset: str = "clinical_trials_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_testing.json"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/alpaca_data_short.json"