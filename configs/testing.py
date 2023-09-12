from dataclasses import dataclass


@dataclass
class test_config:
    batch_size: int = 1
    ft_model_name: str = "out/poc_1_llama-2-7b-hf"
    quantization: bool = True
    dataset: str = "clinical_trials_testing"
    num_workers_dataloader: int = 1
