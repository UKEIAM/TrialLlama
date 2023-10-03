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
# Get currently free cuda device
get_free_cuda_device()

for param_set in param_combinations:
    # Unpack parameter set and add parameters and values to the command
    (
        base_model,
        dataset_version,
        test_dataset_version,
        x_shot_examples,
        dataset_size,
        num_epochs,
        lr,
        temperature,
        top_k,
        top_p,
    ) = param_set

    decimal_part_lr = str(lr).split(".")[1]

    # TODO: Rethink naming. What is the goal? How do I want to track experiments?
    ft_model = f"{base_model.lower()}-{dataset_size}-{dataset_version}-{num_epochs}-{x_shot_examples}"

    command = [
        "python",
        "main.py",
        "--base_model",
        base_model,
        "--x_shot_examples",
        x_shot_examples,
        "--dataset_version",
        dataset_version,
        "--test_dataset_version",
        test_dataset_version,
        "--dataset_size",
        str(dataset_size),
        "--num_epochs",
        str(num_epochs),
        "--ft_model",
        str(ft_model),
        "--lr",
        str(lr),
        "--temperature",
        str(temperature),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
    ]
    # Check if a model was already trained and only experiment needs to be repeated on re_evaluation
    if os.path.exists(os.path.join("out", ft_model)):
        command.append("--run_training")
        command.append(str(False))

    # Run the model script
    subprocess.run(command)
