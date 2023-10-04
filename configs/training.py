# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass


@dataclass
class train_config:
    base_model: str = "Llama-2-13b-chat-hf"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    gradient_accumulation_steps: int = 1
    num_epochs: int = 4
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = True
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset: str = "ct_train_sample_v2"
    dataset_version: str = "v2"
    dataset_size: int = 900
    peft_method: str = "lora"  # None , llama_adapter, prefix
    use_peft: bool = True
    ft_model: str = "llama-2-13b-chat-hf-default"  # Name for the output ft-model
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True
    merge_weights: bool = False
    dist_checkpoint_root_folder: str = "out"  # will be used if using FSDP
    dist_checkpoint_folder: str = "ft-model"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    # max_words: int = 1900 # Since the input is so long, the output of the model while in eval mode is even longer. Hence, we need to restrict the preds output.
    max_tokens: int = 2048  # Fixed: Training does not require that much tokens. Value is set high, since one-/few-shot learning requires prompt + example, which can exceed the token number
    debug: bool = False
