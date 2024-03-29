from dataclasses import dataclass


@dataclass
class test_config:
    batch_size: int = 1
    base_model: str = "Llama-2-13b-chat-hf"
    ft_model: str = "llama-2-13b-chat-hf-default"
    quantization: bool = True
    dataset_version: str = "v7"
    dataset: str = "ct_test_sample_v7"
    dataset_size: int = 100
    num_workers_dataloader: int = 1
    max_new_tokens: int = 600
    seed: int = 42  # seed value for reproducibility
    do_sample: bool = (
        False  # Whether or not to use sampling ; use greedy decoding otherwise.
    )
    min_length: int = None  # The minimum length of the sequence to be generated input prompt + min_new_tokens
    use_cache: bool = True  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0  # [optional] If set to float < 1 only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 0.0
    top_k: int = 50  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = (
        1.0  # The parameter for repetition penalty. 1.0 means no penalty.
    )
    length_penalty: int = 1  # [optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int = (
        None  # the max padding length to be used with tokenizer padding the prompts.
    )
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers make use Flash Attention and Xformer memory-efficient kernels
    load_peft_model: bool = False
    debug: bool = False
    evaluate_base_model: bool = False
    add_example: bool = False
    binary_balancing: bool = True
    topic_year: int = None
    create_dataset_sample: bool = True
