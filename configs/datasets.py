# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class medqa:
    dataset: str = "medqa"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/medical_meadow_medqa.json"


@dataclass
class ct_train_sample_v1:
    dataset: str = "ct_train_sample_v1"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v1.json"


@dataclass
class ct_train_sample_v2:
    dataset: str = "ct_train_sample_v2"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v2.json"


@dataclass
class ct_train_sample_v3:
    dataset: str = "ct_train_sample_v3"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v3.json"


@dataclass
class ct_train_sample_v4:
    dataset: str = "ct_train_sample_v4"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v4.json"


@dataclass
class ct_train_sample_v5:
    dataset: str = "ct_train_sample_v5"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v5.json"


@dataclass
class ct_train_sample_v6:
    dataset: str = "ct_train_sample_v6"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v6.json"


@dataclass
class ct_train_sample_v7:
    dataset: str = "ct_train_sample_v7"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v7.json"


# Test data, never seen by the model
@dataclass
class ct_test_sample_v1:
    dataset: str = "ct_test_sample_v1"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v1.json"


@dataclass
class ct_test_sample_v2:
    dataset: str = "ct_test_sample_v2"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v2.json"


@dataclass
class ct_test_sample_v3:
    dataset: str = "ct_test_sample_v3"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v3.json"


@dataclass
class ct_test_sample_v4:
    dataset: str = "ct_test_sample_v4"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v4.json"


@dataclass
class ct_test_sample_v5:
    dataset: str = "ct_test_sample_v5"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v5.json"


@dataclass
class ct_test_sample_v6:
    dataset: str = "ct_test_sample_v6"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v6.json"


@dataclass
class ct_test_sample_v7:
    dataset: str = "ct_test_sample_v7"
    train_split: str = "train"  # Do not be confused. To utilise the existing code, just passing 'train' as argument returns the whole dataset to the dataloader!
    data_path: str = "data/ct_test_sample_v7.json"
