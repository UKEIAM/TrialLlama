from dataclasses import dataclass


@dataclass
class experiment_config:
    base_model: str = "Llama-2-13b-chat-hf"
    dataset_size: int = 1
    dataset_version: str = "v7"
    dataset_name: str = "ct_train_sample_v7"
    create_sample: bool = True
    dataset_size_testing: int = 50
    dataset_test_version: str = "v8"
    max_tokens: int = 2048
    max_new_tokens: int = 1024
    lr: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 4
    ft_model: str = "llama-2-13b-chat-hf-3"
    load_peft_model: bool = True
    gamma: float = 0.85
    run_training: bool = True
    run_inference: bool = True
    run_eval: bool = True
    top_p: float = 1.0
    temperature: float = 0.1
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: int = 1
    debug: bool = True
    evaluate_base_model: bool = False
    add_example: bool = False
    weight_decay: float = 0.01
    task: str = "classification"
