# TODO: Prepare experiment script

import mlflow
import mlflow.pytorch
import yaml
import os
import logging
import fire


from finetuning import main as ft_main
from testing import main as test_main
from utils.eval_utils import calculate_metrics
from configs.training import train_config
from configs.testing import test_config
from configs.experiments import experiment_config

from utils.config_utils import update_config

base_directory = os.path.dirname(os.path.dirname((__file__)))

# with open(os.path.join(base_directory, "configs", "experiments.yaml"), "r") as file:
#   experiment_config = yaml.safe_load(file)
#
# base_models = experiment_config["base_models"]
# dataset_sozes =  experiment_config["dataset_sizes"]
# learning_rates = experiment_config["lrs"]
# epochs = experiment_config["epochs"]
# batch_sizes = experiment_config["batch_sizes"]
# load_peft_model = experiment_config["load_peft_model"]
# ft_models = experiment_config["ft_models"]


def main(**kwargs):
    update_config((experiment_config), **kwargs)

    dataset = f"ct_{experiment_config.dataset_size}"
    dataset_testing = f"ct_testing_{experiment_config.dataset_size}"
    dataset_path = os.path.join("data", dataset)
    dataset_testing_path = os.path.join(
        "data", f"ct_{experiment_config.dataset_size}_testing.json"
    )
    eval_output_path = os.path.join(
        "out", "eval", f"eval_{experiment_config.ft_model}_qrels.txt"
    )

    qrels_2022_path = os.path.join(
        "data",
        f"trec.nist.gov_data_trials_qrels{experiment_config.gold_labels_year}.txt",
    )

    mlflow.set_experiment(f"{experiment_config.ft_model}")
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "batch_size": experiment_config.batch_size,
                "num_epochs": experiment_config.num_epochs,
                "learning_rate": experiment_config.lr,
                "dataset_size": experiment_config.dataset_size,
                "dataset_name": dataset,
                "qrels_year": experiment_config.gold_labels_year,
                "max_tokens": experiment_config.max_tokens,
                "max_new_tokens": experiment_config.max_new_tokens,
                "temperature": experiment_config.temperature,
                "top_k": experiment_config.top_k,
                "top_p": experiment_config.top_p,
                "length_penalty": experiment_config.length_penalty,
                "repetition_penalty": experiment_config.repetition_penalty,
            }
        )

        if experiment_config.run_training:
            ft_main(
                dataset=dataset,
                lr=experiment_config.lr,
                num_epochs=experiment_config.num_epochs,
                model_name=experiment_config.base_model,
                output_dir=experiment_config.ft_model,
                gamma=experiment_config.gamma,  # TODO: Figure out what Gamma is doing
                max_tokens=experiment_config.max_tokens,
            )
        if experiment_config.run_testing:
            test_main(
                dataset=dataset_testing,
                model_name=experiment_config.base_model,
                ft_model=experiment_config.ft_model,
                load_peft_model=True,
                max_new_tokens=experiment_config.max_new_tokens,
                temperature=experiment_config.temperature,
                top_k=experiment_config.top_k,
                top_p=experiment_config.top_p,
                length_penalty=experiment_config.length_penalty,
                repetition_penalty=experiment_config.repetition_penalty,
            )

        if experiment_config.run_eval:
            scores = calculate_metrics(
                eval_output_path=eval_output_path,
                gold_labels_file=qrels_2022_path,
                ft_model_name=experiment_config.ft_model,
            )
            mlflow.log_metrics(scores)

    # Set experiment ID
    # Run experiments from experiment.yaml
    # Train model on different parameters
    # Run llama_testing.py with fine-tuned model variation
    # Output file with predicted classes
    # Use files to calculate metrics as accuracy, F1 and AUC, etc.


if __name__ == "__main__":
    fire.Fire(main)
