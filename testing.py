import os
import json
import fire
import torch


from peft import PeftModel
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    default_data_collator,
)
from configs.training import train_config
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


def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((test_config), **kwargs)

    clear_gpu_cache()

    create_dataset_sample(dataset_size=test_config.dataset_size, type="test")
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

    max_tokens = get_max_length(
        model
    )  # In the case of running our inference batch evaluation, we have a batch_size of 1, so even with 24gb the max_tokens defined by the model (llama2 4096) is no problem
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

    trec_out, df_out = test(
        model,
        test_set_json,
        test_config,
        test_dataloader,
        tokenizer,
        max_tokens,
    )

    # Save out_file to run with TREC Eval script
    out_dir = os.path.join("out", "eval")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"eval_{test_config.ft_model}_trec.txt")
    out_path_2 = os.path.join(out_dir, f"eval_{test_config.ft_model}_qrels.txt")
    trec_out.to_csv(out_path, sep="\t", index=False, header=False)
    df_out.to_csv(out_path_2, sep="\t", index=False, header=False)
    print(f"Evaluation file for trec_eval script successfully saved under {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
