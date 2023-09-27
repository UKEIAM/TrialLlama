# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os

import pandas as pd

from functools import partial

from ft_datasets.instruct_dataset import InstructionDataset, TestingDataset


DATASET_PREPROC = {
    "ct_v1": partial(InstructionDataset),
    "ct_v2": partial(InstructionDataset),
    "ct_v3": partial(InstructionDataset),
    "ct_v4": partial(InstructionDataset),
    "ct_v5": partial(InstructionDataset),
    "ct_v6": partial(InstructionDataset),
    "ct_testing_v1": partial(TestingDataset),
    "ct_testing_v2": partial(TestingDataset),
    "ct_testing_v3": partial(TestingDataset),
    "ct_testing_v4": partial(TestingDataset),
    "ct_testing_v5": partial(TestingDataset),
    "ct_testing_v6": partial(TestingDataset),
}

base_dir = os.path.dirname(os.path.dirname(__file__))


def get_preprocessed_dataset(
    tokenizer, dataset_config, max_tokens, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
        max_tokens,
    )


def create_dataset_sample(
    dataset_size: int = 300, version: str = "v2", type: str = "train"
) -> None:
    path = base_dir
    out_path = base_dir
    if type == "train":
        path = os.path.join(base_dir, "data", f"ct_full_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_{version}.json")
    elif type == "test":
        path = os.path.join(base_dir, "data", f"ct_testing_full_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_testing_{version}.json")

    df = pd.read_json(path)
    df_filtered = df[
        df["input"].str.contains("Exclusion Criteria")
    ]  # Only take items items into consideration with Inclusion and Exclusion Criteria

    """
        BALANCING: Since "IRRELEVANT" label is predominant in the dataset, we will truncate it in extracting a random
        sample from it based on the average number of items in the two other classes.
        That creates a more balanced, but not perfectly balanced dataset
     """

    col_name = "output"
    value_counts = df[col_name].value_counts()

    max_label = value_counts.index[0]
    avg_item_size_for_truncation = int((value_counts[1] + value_counts[2]) / 2)

    max_label_df = df[df[col_name] == max_label]
    trunc_max_label_df = max_label_df.sample(
        n=avg_item_size_for_truncation, random_state=42, ignore_index=True
    )

    df = df[df[col_name] != max_label]
    balanced_df = pd.concat([df, trunc_max_label_df], ignore_index=True)
    data_sample = balanced_df.sample(n=dataset_size, random_state=42, ignore_index=True)

    data_sample.to_json(out_path, orient="records")
