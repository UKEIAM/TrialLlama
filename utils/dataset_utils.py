# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets.instruct_dataset import (
   InstructionDataset,
   TestingDataset
)

from typing import Optional


DATASET_PREPROC = {
    "clinical_trials_dataset": partial(
        InstructionDataset
    ),  # Adjust max_words based on the requrired input-length input and GPU capacities
    "ct_300": partial(InstructionDataset),
    "ct_10": partial(InstructionDataset),
    "clinical_trials_testing": partial(TestingDataset),
    "ct_testing_300": partial(TestingDataset),
    "ct_testing_10": partial(TestingDataset),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, max_words, split: str = "train"
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
        max_words,
    )
