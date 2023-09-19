# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def merge_weights(base_model: str, peft_model: str, output_dir: str):

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    """
    Fine-tuning with lora outputs adapter weights, which need to be merged with the original model.
    This can be achieved with loading the base-model and then using it as input for the `PeftModel.from_pretrained()` function.
    """
    model = PeftModel.from_pretrained(
        model,
        peft_model,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
