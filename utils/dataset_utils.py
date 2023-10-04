# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import sys
import torch

import pandas as pd

from functools import partial
from typing import Optional

from ft_datasets.instruct_dataset import InstructionDataset, TestingDataset

DATASET_PREPROC = {
    "ct_train_sample_v1": partial(InstructionDataset),
    "ct_train_sample_v2": partial(InstructionDataset),
    "ct_train_sample_v3": partial(InstructionDataset),
    "ct_train_sample_v4": partial(InstructionDataset),
    "ct_test_sample_v1": partial(TestingDataset),
    "ct_test_sample_v2": partial(TestingDataset),
    "ct_test_sample_v3": partial(TestingDataset),
    "ct_test_sample_v4": partial(TestingDataset),
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


def count_words(text):
    if isinstance(text, str):
        return len(text.split())
    else:
        return 0


def create_dataset_sample(
    dataset_size: Optional[int] = None,
    version: str = "v3",
    type: str = "train",
) -> None:
    path = base_dir
    out_path = base_dir

    # TODO: Refactor: Create a sample from ct_all_years and save it as "ct_all_years_testing"
    # Derive train_dataset from "ct_all_years" but leave out ~1000 samples for test
    if type == "train":
        path = os.path.join(base_dir, "data", f"ct_train_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_train_sample_{version}.json")
    elif type == "test":
        path = os.path.join(base_dir, "data", f"ct_test_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_test_sample_{version}.json")

    df = pd.read_json(path)
    df = df[
        df["clinical_trial"].str.contains("Exclusion Criteria:")
    ]  # Only take items  into consideration with Inclusion and Exclusion Criteria

    """
        Some examples are very long. Checking some random samples showed that those are often faulty or just unnecessary
        complex with information. Hence we reduce to 800 words for the input, to provide the model with best possible
        train data.
    """
    cols_to_count = ["instruction", "topic", "clinical_trial", "response"]
    df["word_count"] = df[cols_to_count].apply(
        lambda row: sum(row.map(count_words)), axis=1
    )

    mask = (
        df["word_count"]
        > 1000  # Max tokens of 2048 are somewhat about 1115 words and since a lot of CTs are longer, we go for the save solution.
    )  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
    df = df[~mask]

    df = df.drop(columns=["word_count"])

    if type == "test":
        assert dataset_size <= df.shape[0]
        data_sample = df.sample(n=dataset_size, random_state=42, ignore_index=True)
        data_sample.to_json(out_path, orient="records")
        return

    """
        BALANCING: Since "IRRELEVANT" label is predominant in the dataset, we will truncate it in extracting a random
        sample from it based on the average number of items in the two other classes.
        That creates a more balanced, but not perfectly balanced dataset. Only applied on train data.
        WARNING: Don't change random state value!
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

    if dataset_size == None:
        dataset_size = len(balanced_df)
    assert dataset_size <= balanced_df.shape[0]
    data_sample = balanced_df.sample(n=dataset_size, random_state=42, ignore_index=True)

    data_sample.to_json(out_path, orient="records")
