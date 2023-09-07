from dataclasses import dataclass

@dataclass
class test_config:
  batch_size: int = 1
  ft_model_name: str = 'out/experiment_0_poc'
  quantization: bool = True
  dataset: str = "clinical_trials_testing"
  num_workers_dataloader: int = 1
