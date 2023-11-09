# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import sys
import torch
import math

import pandas as pd

from functools import partial
from typing import Optional

from ft_datasets.instruct_dataset import (
    InstructionDataset,
    TestingDataset,
    QAInstructionDataset,
)

DATASET_PREPROC = {
    "ct_train_sample_v1": partial(InstructionDataset),
    "ct_train_sample_v2": partial(InstructionDataset),
    "ct_train_sample_v3": partial(InstructionDataset),
    "ct_train_sample_v4": partial(InstructionDataset),
    "ct_train_sample_v5": partial(InstructionDataset),
    "ct_train_sample_v5_1": partial(InstructionDataset),
    "ct_train_sample_v5_2": partial(InstructionDataset),
    "ct_train_sample_v6": partial(InstructionDataset),
    "ct_train_sample_v6_1": partial(InstructionDataset),
    "ct_train_sample_v6_2": partial(InstructionDataset),
    "ct_train_sample_v7": partial(InstructionDataset),
    "ct_train_sample_v7_3": partial(InstructionDataset),
    "ct_train_sample_v8": partial(InstructionDataset),
    "ct_train_sample_v8_3": partial(InstructionDataset),
    "ct_train_sample_v9": partial(InstructionDataset),
    "ct_train_sample_v9_3": partial(InstructionDataset),
    "ct_train_sample_v12": partial(InstructionDataset),
    "ct_test_sample_v1": partial(TestingDataset),
    "ct_test_sample_v2": partial(TestingDataset),
    "ct_test_sample_v3": partial(TestingDataset),
    "ct_test_sample_v4": partial(TestingDataset),
    "ct_test_sample_v5": partial(TestingDataset),
    "ct_test_sample_v5_1": partial(InstructionDataset),
    "ct_test_sample_v5_2": partial(InstructionDataset),
    "ct_test_sample_v6": partial(TestingDataset),
    "ct_test_sample_v6_1": partial(InstructionDataset),
    "ct_test_sample_v6_2": partial(InstructionDataset),
    "ct_test_sample_v7": partial(TestingDataset),
    "ct_test_sample_v7_3": partial(TestingDataset),
    "ct_test_sample_v8": partial(TestingDataset),
    "ct_test_sample_v8_3": partial(TestingDataset),
    "ct_test_sample_v9": partial(TestingDataset),
    "ct_test_sample_v9_3": partial(TestingDataset),
    "ct_test_sample_v12": partial(TestingDataset),
    "medqa": partial(QAInstructionDataset),
}

base_dir = os.path.dirname(os.path.dirname(__file__))


def get_preprocessed_dataset(
    tokenizer,
    dataset_config,
    max_tokens,
    split: str = "train",
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
    add_example: Optional[bool] = False,
    version: str = "v7",
    type: str = "train",
    binary_eval: Optional[bool] = False,
) -> None:
    path = base_dir
    out_path = base_dir

    # TODO: Refactor: Create a sample from ct_all_years and save it as "ct_all_years_testing"
    # Derive train_dataset from "ct_all_years" but leave out ~1000 samples for testssh
    if type == "train":
        path = os.path.join(base_dir, "data", f"ct_train_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_train_sample_{version}.json")
    elif type == "test":
        if add_example:
            example_path = os.path.join(
                base_dir, "data", f"inference_example_{version}.json"
            )
        path = os.path.join(base_dir, "data", f"ct_test_{version}.json")
        out_path = os.path.join(base_dir, "data", f"ct_test_sample_{version}.json")

    df = pd.read_json(path)

    """
        Some examples are very long. Checking some random samples showed that those are often faulty or just unnecessary
        complex with information. Hence we reduce to 1000 words for the input, to provide the model with best possible
        train data.
    """
    if type == "test":
        word_count = 1500
    else:
        word_count = 1000
    df = truncate(df, word_count)
    df.drop(["word_count"], axis=1, inplace=True)

    """
        BALANCING: Since "IRRELEVANT" label is predominant in the dataset, we will truncate it in extracting a random
        sample from it based on the average number of items in the two other classes.
        That creates a more balanced, but not perfectly balanced dataset. Only applied on train data.
        WARNING: Don"t change random state value!
    """

    # CURATED DATASET: Reduce amount of data in returning only x examples per patient topic
    df["topic_id"] = df["id"].str.split("_").str[1]
    balanced_df = pd.DataFrame(columns=df.columns)
    if dataset_size is None:
        dataset_size = len(df)

    for unique_id in df["topic_id"].unique():
        # Get all rows with the current unique ID
        id_subset = df[df["topic_id"] == unique_id]
        if dataset_size > 3:
            desired_label_count = (
                id_subset.groupby("topic_id")["output"]
                .value_counts()
                .sort_values()
                .iloc[0]
            )
        else:
            desired_label_count = dataset_size
        # Separate the rows by label
        label_groups = [
            id_subset[id_subset["output"] == label]
            for label in id_subset["output"].unique()
        ]

        # Balance each label group to have the desired_label_count
        if binary_eval:
            try:
                # Sometimes the aspired split is not possible since not enough "eligible" labels are available. Hence we do the following checks.
                aspired_total_sampels = desired_label_count * 3
                len_eligible = len(id_subset[id_subset["output"] == "A: eligible"])
                eligible_samples = (
                    len_eligible
                    if len_eligible < math.ceil(aspired_total_sampels / 2)
                    else math.ceil(aspired_total_sampels / 2)
                )
                eligible = id_subset[id_subset["output"] == "A: eligible"].sample(
                    eligible_samples
                )

                len_excluded = len(id_subset[id_subset["output"] == "B: excluded"])
                excluded_samples = (
                    len_excluded
                    if len_excluded < math.ceil(aspired_total_sampels / 4)
                    else math.ceil(aspired_total_sampels / 4)
                )
                excluded = id_subset[id_subset["output"] == "B: excluded"].sample(
                    excluded_samples
                )

                len_irrelevant = len(id_subset[id_subset["output"] == "C: irrelevant"])
                irrelevant_sample = (
                    len_irrelevant
                    if len_irrelevant < math.ceil(aspired_total_sampels / 4)
                    else math.ceil(aspired_total_sampels / 4)
                )
                irrelevant = id_subset[id_subset["output"] == "C: irrelevant"].sample(
                    irrelevant_sample
                )

                balanced_df = pd.concat(
                    [balanced_df, eligible, excluded, irrelevant], ignore_index=True
                )
            except ValueError as e:
                print("VALUE COUNTS 0: One of the grouped labels is 0. Continuing.")
                continue
        else:
            balanced_label_groups = [
                group.sample(n=len(group), random_state=42)
                if len(group) < desired_label_count
                else group.sample(n=desired_label_count, random_state=42)
                for group in label_groups
            ]

            # Concatenate the balanced label groups and add them to the balanced_df
            for df_item in balanced_label_groups:
                balanced_df = pd.concat([balanced_df, df_item], ignore_index=True)

    # balanced_df.drop(["topic_id"], axis=1, inplace=True)

    if type == "test":
        if add_example:
            df_example = pd.read_json(example_path)
            input_value = df_example["input"]
            balanced_df["instruction"] = balanced_df["instruction"] + input_value[0]
        balanced_df = truncate(balanced_df, word_count)
        balanced_df.drop(["word_count"], axis=1, inplace=True)

    samples = balanced_df.shape[0]
    if dataset_size > 3:
        try:
            assert dataset_size <= balanced_df.shape[0]
            samples = dataset_size
        except AssertionError:
            print(
                "WARNING: Balanced data set smaller than desired data set size. Returning maximum data set size."
            )
            samples = balanced_df.shape[0]

    data_sample = balanced_df.sample(n=samples, random_state=42, ignore_index=True)
    data_sample.to_json(out_path, orient="records")


def truncate(df, word_count):
    cols_to_count = ["instruction", "topic", "clinical_trial", "response"]
    df["word_count"] = df[cols_to_count].apply(
        lambda row: sum(row.map(count_words)), axis=1
    )
    mask = (
        df["word_count"]
        > word_count
        # Max tokens of 2048 are somewhat about 1115 words and since a lot of CTs are longer, we go for the save solution.
    )  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
    df = df[~mask]

    return df
