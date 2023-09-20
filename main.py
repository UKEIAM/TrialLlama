import yaml
import itertools
import subprocess
import argparse

# Load parameters from the YAML file
with open("configs/experiment_definitions.yaml", "r") as yaml_file:
    parameters = yaml.safe_load(yaml_file)

# Generate all combinations of parameters
param_combinations = list(itertools.product(*parameters.values()))

# Loop through each combination and run your model script
for param_set in param_combinations:
    # Unpack parameter set and add parameters and values to the command
    base_model, dataset_size, num_epochs = param_set
    ft_model = (
        f'{base_model.replace("-hf", "").replace("-2", "").lower()}-{dataset_size}'
    )
    command = [
        "python",
        "run_experiment.py",
        "--base_model",
        base_model,
        "--num_epochs",
        num_epochs,
        "--dataset_size",
        str(dataset_size),
        "--ft_model",
        ft_model,
    ]

    print(command)
    # Run the model script
    subprocess.run(command)
