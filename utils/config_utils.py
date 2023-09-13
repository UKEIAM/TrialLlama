# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import fields
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)

import configs.datasets as datasets
from configs.peft import lora_config, llama_adapter_config, prefix_config
from configs.training import train_config
from configs.testing import test_config
from .dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")
            elif isinstance(config, test_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(config, kwargs):
    configs = (lora_config, llama_adapter_config, prefix_config)
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    assert (
        config.peft_method in names
    ), f"Peft config not found: {config.peft_method}"

    config = configs[names.index(config.peft_method)]
    update_config(config, **kwargs)
    params = {k.name: getattr(config, k.name) for k in fields(config)}
    peft_config = peft_configs[names.index(config.peft_method)](**params)

    return peft_config


def generate_dataset_config(config, kwargs):
    names = tuple(DATASET_PREPROC.keys())

    assert config.dataset in names, f"Unknown dataset: {config.dataset}"

    dataset_config = {k: v for k, v in inspect.getmembers(datasets)}[
        config.dataset
    ]
    update_config(dataset_config, **kwargs)

    return dataset_config
