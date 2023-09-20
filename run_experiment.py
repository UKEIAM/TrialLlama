# TODO: Prepare experiment script

import mlflow
import mlflow.pytorch
import os
import logging
import fire


from finetuning import main as ft_main
from testing import main as test_main
from utils.eval_utils import calculate_metrics
from configs.experiments import experiment_config
from utils.config_utils import update_config

base_directory = os.path.dirname(os.path.dirname((__file__)))


def main(**kwargs):
    update_config((experiment_config), **kwargs)

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
                "dataset_size_testing": experiment_config.dataset_size_testing,
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
            results = ft_main(
                dataset_size=experiment_config.dataset_size,
                lr=experiment_config.lr,
                num_epochs=experiment_config.num_epochs,
                model_name=experiment_config.base_model,
                output_dir=experiment_config.ft_model,
                gamma=experiment_config.gamma,  # TODO: Figure out what Gamma is doing
                max_tokens=experiment_config.max_tokens,
            )
            mlflow.log_metrics(results)

        if experiment_config.run_testing:
            results = test_main(
                dataset_size=experiment_config.dataset_size_testing,
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
            mlflow.log_metrics(results)

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
