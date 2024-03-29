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
# TODO: setting free cuda device is not working in certain cases e.g. when all devices are currently busy (even only partly)
# gpu_available = get_free_cuda_device()
# if not gpu_available:
#     print("No GPU available. Exiting.")
#     exit()

for param_set in param_combinations:
    # Unpack parameter set and add parameters and values to the command
    (
        base_model,
        dataset_version,
        dataset_test_version,
        dataset_size,
        num_epochs,
        grad_acc,
        lr,
        weight_decay,
        temperature,
        evaluate_base_model,
        task,
        binary_balancing,
        one_shot,
    ) = param_set

    # decimal_part_lr = str(lr).split(".")[1] if "." in str(lr) else "0"

    # TODO: Check if learning rate at the end works, since dot in folder name
    model_version = "v5"
    if evaluate_base_model:
        ft_model = f"{base_model.lower()}-base"
    else:
        ft_model = f"{base_model.lower()}-{dataset_size}-{dataset_version}-{num_epochs}-{model_version}-{lr}-{grad_acc}"

    if task == "reasoning":
        one_shot = True
        binary_balancing = False

    if task == "classification":
        max_new_tokens = 10
        dataset_size_testing = None
    else:
        max_new_tokens = 1000
        dataset_size_testing = 20

    command = [
        "python",
        "main.py",
        "--base_model",
        base_model,
        "--dataset_version",
        dataset_version,
        "--dataset_test_version",
        dataset_test_version,
        "--dataset_size",
        str(dataset_size),
        "--num_epochs",
        str(num_epochs),
        "--ft_model",
        str(ft_model),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--temperature",
        str(temperature),
        "--evaluate_base_model",
        str(evaluate_base_model),
        "--max_new_tokens",
        str(max_new_tokens),
        "--task",
        str(task),
        "--binary_balancing",
        str(binary_balancing),
        "--gradient_accumulation_steps",
        str(grad_acc),
        "--dataset_size_testing",
        str(dataset_size_testing),
    ]
    # Check if a model was already trained and only experiment needs to be repeated on re_evaluation
    if os.path.exists(os.path.join("out", ft_model)):
        command.append("--run_training")
        command.append(str(False))

    # For response generation we add a one-shot example to enhance the output quality of the model
    if one_shot:
        command.append("--add_example")
        command.append(str(True))

    if evaluate_base_model:
        command.append("--load_peft_model")
        command.append(str(False))
        command.append("--run_training")
        command.append(str(False))
        print("EVALUATING BASE MODEL")

    # Run the model script
    subprocess.run(command)
