# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


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
class ct_train_sample_v5_1:
    dataset: str = "ct_train_sample_v5_1"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v5_1.json"


@dataclass
class ct_train_sample_v5_2:
    dataset: str = "ct_train_sample_v5_2"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v5_2.json"


@dataclass
class ct_train_sample_v6:
    dataset: str = "ct_train_sample_v6"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v6.json"


@dataclass
class ct_train_sample_v6_1:
    dataset: str = "ct_train_sample_v6_1"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v6_1.json"


@dataclass
class ct_train_sample_v6_2:
    dataset: str = "ct_train_sample_v6_2"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v6_2.json"


@dataclass
class ct_train_sample_v7:
    dataset: str = "ct_train_sample_v7"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v7.json"


@dataclass
class ct_train_sample_v7_3:
    dataset: str = "ct_train_sample_v7_3"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v7_3.json"


@dataclass
class ct_train_sample_v8:
    dataset: str = "ct_train_sample_v8"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v8.json"


@dataclass
class ct_train_sample_v8_3:
    dataset: str = "ct_train_sample_v8_3"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v8_3.json"


@dataclass
class ct_train_sample_v9:
    dataset: str = "ct_train_sample_v9"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v9.json"


@dataclass
class ct_train_sample_v9_3:
    dataset: str = "ct_train_sample_v9_3"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v9_3.json"


@dataclass
class ct_train_sample_v12:
    dataset: str = "ct_train_sample_v12"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "data/ct_train_sample_v12.json"


# Test data, never seen by the model
@dataclass
class ct_test_sample_v1:
    dataset: str = "ct_test_sample_v1"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v1.json"


@dataclass
class ct_test_sample_v2:
    dataset: str = "ct_test_sample_v2"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v2.json"


@dataclass
class ct_test_sample_v3:
    dataset: str = "ct_test_sample_v3"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v3.json"


@dataclass
class ct_test_sample_v4:
    dataset: str = "ct_test_sample_v4"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v4.json"


@dataclass
class ct_test_sample_v5:
    dataset: str = "ct_test_sample_v5"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v5.json"


@dataclass
class ct_test_sample_v5_1:
    dataset: str = "ct_test_sample_v5_1"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v5_1.json"


@dataclass
class ct_test_sample_v5_2:
    dataset: str = "ct_test_sample_v5_2"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v5_2.json"


@dataclass
class ct_test_sample_v6:
    dataset: str = "ct_test_sample_v6"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v6.json"


@dataclass
class ct_test_sample_v6_1:
    dataset: str = "ct_test_sample_v6_1"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v6_1.json"


@dataclass
class ct_test_sample_v6_2:
    dataset: str = "ct_test_sample_v6_2"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v6_2.json"


@dataclass
class ct_test_sample_v7:
    dataset: str = "ct_test_sample_v7"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v7.json"


@dataclass
class ct_test_sample_v7_3:
    dataset: str = "ct_test_sample_v7_3"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v7_3.json"


@dataclass
class ct_test_sample_v8:
    dataset: str = "ct_test_sample_v8"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v8.json"


@dataclass
class ct_test_sample_v8_3:
    dataset: str = "ct_test_sample_v8_3"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v8_3json"


@dataclass
class ct_test_sample_v9:
    dataset: str = "ct_test_sample_v9"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v9.json"


@dataclass
class ct_test_sample_v9_3:
    dataset: str = "ct_test_sample_v9_3"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v9_3.json"


@dataclass
class ct_test_sample_v12:
    dataset: str = "ct_test_sample_v12"
    test_split: str = "test"
    data_path: str = "data/ct_test_sample_v12.json"
