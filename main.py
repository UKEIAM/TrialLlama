import yaml
import itertools
import subprocess
import argparse

from utils.set_free_cuda_device import get_free_cuda_device

# Load parameters from the YAML file
with open("configs/experiment_definitions.yaml", "r") as yaml_file:
    parameters = yaml.safe_load(yaml_file)

# Generate all combinations of parameters
param_combinations = list(itertools.product(*parameters.values()))
get_free_cuda_device()
# Loop through each combination and run your model script
# Kind of something similar to Hydra :D
for param_set in param_combinations:
    # Unpack parameter set and add parameters and values to the command
    base_model, dataset_size, num_epochs, lr, temperature = param_set
    ft_model = (
        f'{base_model.replace("-hf", "").replace("-2", "").lower()}-{dataset_size}'
    )
    command = [
        "python",
        "run_experiment.py",
        f"--base_model {base_model}",
        f"--num_epochs {num_epochs}",
        f"--dataset_size {dataset_size}",
        f"--ft_model {ft_model}",
        f"--lr {lr}",
        f"--temperature {temperature}",
    ]

    # Run the model script
    subprocess.run(command)
