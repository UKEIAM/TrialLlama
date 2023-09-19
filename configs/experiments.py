from dataclasses import dataclass


@dataclass
class experiment_config:
    base_model: str = "Llama-2-7b-chat-hf"
    base_models: list = [
        "Llama-2-13b-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-7b-hf",
        "Llama-2-7b-chat-hf",
    ]
    dataset_size: int = 300
    dataset_sizes: list = [25000, 10000, 5000, 1800, 900, 500, 300, 10]
    max_tokens: int = 2048
    max_new_tokens: int = 20
    lr: str = "1e-4"
    lrs: list = ["1e-3", "1e-4", "1e-5", "1e-6"]
    batch_size: int = 4
    batch_sizes: list = [2, 3, 4]
    num_epochs: int = 3
    epochs: list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ft_model: str = "llama-7b-chat-300"
    ft_models: list = [
        "llama-7b-chat-300",
        "llama-7b-chat-500",
        "llama-7b-chat-900",
        "llama-7b-chat-1800",
    ]
    gold_labels_year: int = 2022
    gold_labels_21: str = "trec.nist.gov_data_trials_qrels2021.txt"
    gold_labels_22: str = "trec.nist.gov_data_trials_qrels2022.txt"
    device_id: int = 3
    gamma: float = 0.85
    run_training: bool = True
    run_testing: bool = True
    run_eval: bool = True
