import os
import json

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import PeftModel
from pkg_resources import packaging
from tqdm import tqdm
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from transformers import (
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)

from inference.model_utils import load_model, load_peft_model

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
        max_words=test_config.max_tokens,
        split="train",
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
    model = load_model(test_config.ft_model_name, test_config.quantization)
    if test_config.load_peft_model:
        model = load_peft_model(model, test_config.ft_model_name + "adapter_weights")

    model.eval()
    model.resize_token_embeddings(model.config.vocab_size + 1)
    # test_prompt = "This is a test sentence to tokenize."
    # batch_plain = tokenizer(
    #     test_prompt,
    #     padding="max_length",
    #     truncation=True,
    #     max_length=test_config.max_tokens,
    #     return_tensors="pt",
    #     )
    
    # batch_plain = {k: v.to("cuda") for k, v in batch_plain.items()}

    trec_out, df_out = test(
        model,
        test_set_json,
        test_config,
        test_dataloader,
        tokenizer,
    )

    # Save out_file to run with TREC Eval script
    out_path = os.path.join("out", "eval", f"{test_config.out_file_name}_trec.txt")
    out_path_2 = os.path.join("out", "eval", f"{test_config.out_file_name}_qrels.txt")
    trec_out.to_csv(out_path, sep="\t", index=False, header=False)
    df_out.to_csv(out_path_2, sep="\t", index=False, header=False)
    print(f"Evaluation file for trec_eval script successfully saved under {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
