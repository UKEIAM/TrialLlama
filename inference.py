import os
import json
import fire
import torch
import mlflow

from peft import PeftModel
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)
from configs.testing import test_config
from utils.config_utils import (
    update_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset, create_dataset_sample
from utils.train_utils import (
    clear_gpu_cache,
)
from utils.test_utils import test, get_max_length
from typing import Optional


def main(
    eval_output_path: Optional[str] = None, logger: Optional[object] = None, **kwargs
):
    # Update the configuration for the training and sharding process
    update_config((test_config), **kwargs)

    clear_gpu_cache()

    create_dataset_sample(
        dataset_size=test_config.dataset_size,
        version=test_config.dataset_version,
        type="test",
    )
    dataset_config = generate_dataset_config(test_config, kwargs)

    # Load fine-tuned model ATTENTION: Fine-tuned adapter weights, need to be merged with base-model before loading is possible!
    if test_config.load_peft_model:
        model_path = os.path.join("checkpoints", "meta-llama", test_config.base_model)
        base_model = LlamaForCausalLM.from_pretrained(
            model_path,
            return_dict=True,
            load_in_8bit=test_config.quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        ft_model_path = os.path.join("out", test_config.ft_model, "adapter_weights")
        model = PeftModel.from_pretrained(
            base_model,
            ft_model_path,
        )
    elif test_config.load_base_model:
        model = LlamaForCausalLM.from_pretrained(
            os.path.join("checkpints", "meta-llama", "Llama-2-13b-chat-hf"),
            return_dict=True,
            load_in_8bit=test_config.quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            os.path.join("out", test_config.ft_model),
            return_dict=True,
            load_in_8bit=test_config.quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join("checkpoints", "meta-llama", test_config.base_model)
    )
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    max_tokens = 3072  # Max tokens need to be set beneath the max_tokens the model might be able to ingest, since the response otherwise exceeds the number of max_tokens

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        max_tokens=max_tokens,
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

    model.eval()
    model.resize_token_embeddings(model.config.vocab_size + 1)

    raw_out, erc = test(
        model,
        test_set_json,
        test_config,
        test_dataloader,
        tokenizer,
        max_tokens,
        logger=logger,
    )

    # Save out_file to run with TREC Eval script
    out_dir = os.path.join("out", "eval")
    os.makedirs(out_dir, exist_ok=True)

    if eval_output_path is None:
        eval_output_path = os.path.join(out_dir, f"eval_{test_config.ft_model}.json")

    raw_out.to_json(eval_output_path, orient="records")
    print(f"Evaluation file successfully saved under {out_dir}")

    return erc


if __name__ == "__main__":
    fire.Fire(main)
