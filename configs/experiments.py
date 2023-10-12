from dataclasses import dataclass


@dataclass
class experiment_config:
    base_model: str = "Llama-2-13b-chat-hf"
    dataset_size: int = 1
    dataset_version: str = "v3"
    dataset_name: str = f"ct_train_sample_{dataset_version}"
    create_sample: bool = (True,)
    dataset_size_testing: int = 20
    test_dataset_version: str = "v3"
    max_tokens: int = 2048
    max_new_tokens: int = 200
    lr: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 4
    ft_model: str = "llama-2-13b-chat-hf-300-default"
    gold_labels_year: int = 2022
    gamma: float = 0.85
    run_training: bool = True
    run_inference: bool = True
    run_eval: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: int = 1
    debug: bool = True
    evaluate_base_model: bool = False
