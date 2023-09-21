import yaml
import itertools
import subprocess
import os

from utils.set_free_cuda_device import get_free_cuda_device

# Load parameters from the YAML file
with open("configs/experiment_definitions.yaml", "r") as yaml_file:
    parameters = yaml.safe_load(yaml_file)

# Generate all combinations of parameters
param_combinations = list(itertools.product(*parameters.values()))
# Loop through each combination and run your model script
# Kind of something similar to Hydra :D
for param_set in param_combinations:
    # Unpack parameter set and add parameters and values to the command
    base_model, dataset_size, num_epochs, lr, temperature, top_k, top_p = param_set
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
        f"--top_k {top_k}",
        f"--top_p {top_p}",
    ]

    # Check if a model was already trained and only experiment needs to be repeated on re_evaluation
    if os.path.exists(os.path.join("out", ft_model)):
        command.append("--run_train False")

    # Get currently free cuda device
    # TODO: Check out if this position works, putting get_free_cuda_device() to run_experiment.py did not work...
    get_free_cuda_device()

    # Run the model script
    subprocess.run(command)
