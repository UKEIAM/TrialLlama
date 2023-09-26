# TODO: Prepare experiment script
import os
import logging
import fire

import mlflow
import mlflow.pytorch

from utils.train_utils import clear_gpu_cache
from finetuning import main as ft_main
from testing import main as test_main
from utils.eval_utils import calculate_metrics
from configs.experiments import experiment_config
from utils.config_utils import update_config
from utils.logger_utils import setup_logger
from utils.set_free_cuda_device import get_free_cuda_device

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

    mlflow.set_experiment(
        f"{experiment_config.ft_model}_{experiment_config.experiment_focus}"
    )
    description = f"Fine-tuned model {experiment_config.ft_model} | batch-size of {experiment_config.batch_size} | number of epochs of {experiment_config.num_epochs} | lr of {experiment_config.lr} | qrels {experiment_config.gold_labels_year}"
    with mlflow.start_run(
        description=description,
        run_name=f"ft-llama-{experiment_config.temperature}-{experiment_config.top_k}-{experiment_config.top_p}-{experiment_config}",
    ) as run:
        logger = setup_logger(run_id=run.info.run_id)
        mlflow.log_params(
            {
                "dataset_version": experiment_config.dataset_version,
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
                "run_training": experiment_config.run_training,
                "run_testing": experiment_config.run_testing,
                "run_eval": experiment_config.run_eval,
            }
        )

        if experiment_config.run_training:
            print("Running training...")
            results = ft_main(
                logger=logger,
                dataset_size=experiment_config.dataset_size,
                lr=experiment_config.lr,
                num_epochs=experiment_config.num_epochs,
                model_name=experiment_config.base_model,
                ft_model=experiment_config.ft_model,
                gamma=experiment_config.gamma,  # TODO: Figure out what Gamma is doing
                max_tokens=experiment_config.max_tokens,
            )
            mlflow.set_tag("ft-conducted", "TRUE")
            mlflow.log_metrics(results)
            clear_gpu_cache()

        if experiment_config.run_testing:
            print("Running testing...")
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
                debug=experiment_config.debug,
                logger=logger,
            )
            mlflow.log_metric("number_of_empty_responses", results)
            clear_gpu_cache()

        if experiment_config.run_eval:
            print("Running evaluation...")
            run_name = run.info.run_name
            scores = calculate_metrics(
                eval_output_path=eval_output_path,
                gold_labels_file=qrels_2022_path,
                ft_model_name=experiment_config.ft_model,
                run_name=run_name,
                logger=logger,
            )
            mlflow.log_metrics(scores)
            clear_gpu_cache()

        # Clean gpu_cache before next mlflow run
        clear_gpu_cache()
        print(f"Run with ID {run.info.run_id} finished successful")
    # Set experiment ID
    # Run experiments from experiment.yaml
    # Train model on different parameters
    # Run llama_testing.py with fine-tuned model variation
    # Output file with predicted classes
    # Use files to calculate metrics as accuracy, F1 and AUC, etc.


if __name__ == "__main__":
    fire.Fire(main)
