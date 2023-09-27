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
    # Split the text into words using whitespace as a separator and count them
    words = text.split()
    return len(words)


def create_dataset_sample(
    dataset_size: int = 300,
    version: str = "v2",
    type: str = "train",
    x_shot_examples: Optional[str] = "",
    logger: Optional[object] = None,
) -> None:
    path = base_dir
    out_path = base_dir
    x_shot_examples_path = None
    if x_shot_examples == "few-shot":
        x_shot_examples_path = os.path.join(
            base_dir, "data", f"ct_few_shot_{version}.json"
        )
    elif x_shot_examples == "one-shot":
        x_shot_examples_path = os.path.join(
            base_dir, "data", f"ct_one_shot_{version}.json"
        )

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
        df["input"].str.contains("Exclusion Criteria:")
    ]  # Only take items  into consideration with Inclusion and Exclusion Criteria
    """
        Some randomly picked examples are prepared to create hand-crafted few-shot and one-shot learning examples.
        Those examples should be excluded while sampling and always added to the top of the retrieved data sample.
    """
    nr_examples = 0
    if x_shot_examples != None:
        df_examples = pd.read_json(x_shot_examples_path)
        nr_examples = len(df_examples)
        blacklist = [
            (x, y) for x, y in zip(df_examples["id"].values, df_examples["year"].values)
        ]
        df = df[~df.set_index(["id", "year"]).index.isin(blacklist)]

    """
        Some examples are very long. Checking some random samples showed that those are often faulty or just unnecessary
        complex with information. Hence we reduce to 600 words for the input, to provide the model with best possible
        train data.
    """
    df["word_count"] = df["input"].apply(count_words)

    mask = (
        df["word_count"] > 600
    )  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
    df = df[~mask]

    df = df.drop(columns=["word_count"])

    """
        BALANCING: Since "IRRELEVANT" label is predominant in the dataset, we will truncate it in extracting a random
        sample from it based on the average number of items in the two other classes.
        That creates a more balanced, but not perfectly balanced dataset
     """

    col_name = "output"
    value_counts = df[col_name].value_counts()

    max_label = value_counts.index[0]
    avg_item_size_for_truncation = int((value_counts[1] + value_counts[2]) / 2)

    """
        WARNING: Don't change random state. To enable few-shot learning first x samples are modified by hand to give the
        model a starting point.
    """
    max_label_df = df[df[col_name] == max_label]
    trunc_max_label_df = max_label_df.sample(
        n=avg_item_size_for_truncation, random_state=42, ignore_index=True
    )

    df = df[df[col_name] != max_label]
    balanced_df = pd.concat([df, trunc_max_label_df], ignore_index=True)
    try:
        data_sample = balanced_df.sample(
            n=(dataset_size - nr_examples), random_state=42, ignore_index=True
        )
    except Exception as e:
        logger.error(f"DATASET SAMPLE CREATION FAILED with error: {e}")
        sys.exit(1)

    if nr_examples > 0:
        data_sample = pd.concat(df_examples, data_sample)

    data_sample.to_json(out_path, orient="records")
