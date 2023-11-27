import os
import json
import fire
import torch

from peft import PeftModel
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    default_data_collator,
)

from inference.model_utils import load_model, load_peft_model
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
        add_example=test_config.add_example,
        version=test_config.dataset_version,
        type="test",
        binary_balancing=test_config.binary_balancing,
    )
    dataset_config = generate_dataset_config(test_config, kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(test_config.seed)
    torch.manual_seed(test_config.seed)
    base_model_path = os.path.join("checkpoints", "meta-llama", test_config.base_model)
    ft_model_path = os.path.join("out", test_config.ft_model, "adapter_weights")
    model = load_model(base_model_path, test_config.quantization)

    if test_config.load_peft_model:
        print("LOADING PEFT MODEL")
        model = load_peft_model(model, ft_model_path)

    model.eval()

    # After updating all libraries, LlamaTokenizer throws error when trying to load weights. AutoTokenizer works.
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    max_tokens = 3072

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
        eval_output_path = os.path.join(
            out_dir, f"eval_{test_config.ft_model}_{test_config.dataset_version}.json"
        )

    raw_out.to_json(eval_output_path, orient="records")
    print(f"Evaluation file successfully saved under {out_dir}")

    return erc


if __name__ == "__main__":
    fire.Fire(main)
