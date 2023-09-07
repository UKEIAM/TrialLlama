import os
import json

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import PeftModel
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from configs.training import train_config
from configs.testing import test_config
from policies.anyprecision_optimizer import AnyPrecisionAdamW

from utils.fsdp_utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    clear_gpu_cache,
)

from utils.test_utils import test


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((test_config), **kwargs)
    clear_gpu_cache()

    # Initialise dataloader, NO shuffling!
    tokenizer = LlamaTokenizer.from_pretrained(test_config.ft_model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    dataset_config = generate_dataset_config(test_config, kwargs)

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",  # Not to be confused: Simply used existing infrastructer, to load full dataset provided
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=test_config.batch_size,
        num_workers=test_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    # Load JSON for mapping
    test_set_json = json.load(open(dataset_config.data_path))

    # Load fine-tuned model ATTENTION: Fine-tuned adapter weights, need to be merged with base-model before loading is possible!
    model = LlamaForCausalLM.from_pretrained(
        test_config.ft_model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    test(
        model,
        test_set_json,
        test_dataloader,
        tokenizer,
        test_config.gradient_accumulation_steps,
        test_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
