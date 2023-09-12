from dataclasses import dataclass


@dataclass
class test_config:
    batch_size: int = 1
    ft_model_name: str = "out/poc_1_llama-2-7b-hf"
    quantization: bool = True
    dataset: str = "ct_testing_10" # TODO: Change to full dataset, once debugging finished
    num_workers_dataloader: int = 1
    out_file_name: str = "poc_1_eval_ready"
