from dataclasses import dataclass


@dataclass
class experiment_config:
    base_model: str = "Llama-2-7b-chat-hf"
    dataset_size: int = 300
    max_tokens: int = 2048
    max_new_tokens: int = 20
    lr: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    ft_model: str = "llama-7b-chat-300"
    gold_labels_year: int = 2022
    gamma: float = 0.85
    run_training: bool = True
    run_testing: bool = True
    run_eval: bool = True
