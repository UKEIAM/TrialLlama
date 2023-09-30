# TODO: Prepare experiment script
import os
import logging
import fire

import mlflow
import mlflow.pytorch

from utils.train_utils import clear_gpu_cache
from finetuning import main as ft_main
from inference import main as test_main
from utils.eval_utils import calculate_metrics, prepare_files
from configs.experiments import experiment_config
from utils.config_utils import update_config
from utils.logger_utils import setup_logger
from utils.set_free_cuda_device import get_free_cuda_device

base_directory = os.path.dirname(os.path.dirname((__file__)))


def main(**kwargs):
    update_config((experiment_config), **kwargs)

    qrels_2022_path = os.path.join(
        "data",
        f"trec.nist.gov_data_trials_qrels{experiment_config.gold_labels_year}.txt",
    )

    if experiment_config.x_shot_examples == "few-shot":
        x_shot_examples_path = os.path.join(
            base_directory,
            "data",
            f"ct_few_shot_{experiment_config.dataset_version}.json",
        )
    elif experiment_config.x_shot_examples == "one-shot":
        x_shot_examples_path = os.path.join(
            base_directory,
            "data",
            f"ct_one_shot_{experiment_config.dataset_version}.json",
        )

    mlflow.set_experiment(f"{experiment_config.ft_model}")
    description = f"Fine-tuned model {experiment_config.ft_model} | qrels {experiment_config.gold_labels_year}"
    with mlflow.start_run(
        description=description,
    ) as run:
        logger = setup_logger(run_id=run.info.run_id)
        run_name = run.info.run_name
        raw_eval_output_path = os.path.join(
            "out",
            "eval",
            f"eval_{experiment_config.ft_model}_{run_name}_{experiment_config.test_dataset_version}_raw.json",
        )
        eval_output_path = os.path.join(
            "out",
            "eval",
            f"eval_{experiment_config.ft_model}_{run_name}_{experiment_config.test_dataset_version}.json",
        )
        trec_eval_output_path = os.path.join(
            "out",
            "eval",
            f"eval_{experiment_config.ft_model}_{run_name}_{experiment_config.test_dataset_version}_trec.txt",
        )
        mlflow.log_params(
            {
                "batch_size": experiment_config.batch_size,
                "num_epochs": experiment_config.num_epochs,
                "learning_rate": experiment_config.lr,
                "dataset_version": experiment_config.dataset_version,
                "test_dataset_version": experiment_config.test_dataset_version,
                "dataset_size": experiment_config.dataset_size,
                "dataset_size_testing": experiment_config.dataset_size_testing,
                "x_shot_examples": experiment_config.x_shot_examples,
                "qrels_year": experiment_config.gold_labels_year,
                "max_tokens": experiment_config.max_tokens,
                "max_new_tokens": experiment_config.max_new_tokens,
                "temperature": experiment_config.temperature,
                "top_k": experiment_config.top_k,
                "top_p": experiment_config.top_p,
                "length_penalty": experiment_config.length_penalty,
                "repetition_penalty": experiment_config.repetition_penalty,
                "run_training": experiment_config.run_training,
                "run_inference": experiment_config.run_inference,
                "run_eval": experiment_config.run_eval,
            }
        )

        if experiment_config.run_training:
            print("Running training...")
            results = ft_main(
                logger=logger,
                dataset_version=experiment_config.dataset_version,
                dataset=f"ct_train_sample_{experiment_config.dataset_version}",
                dataset_size=experiment_config.dataset_size,
                x_shot_example=experiment_config.x_shot_examples,
                lr=experiment_config.lr,
                num_epochs=experiment_config.num_epochs,
                model_name=experiment_config.base_model,
                ft_model=experiment_config.ft_model,
                max_tokens=experiment_config.max_tokens,
            )
            mlflow.set_tag("ft_conducted", "TRUE")
            mlflow.log_metrics(results)
            clear_gpu_cache()

        if experiment_config.run_inference:
            print("Running testing...")
            results = test_main(
                dataset_size=experiment_config.dataset_size_testing,
                dataset_version=experiment_config.dataset_version,
                dataset=f"ct_test_sample_{experiment_config.dataset_version}",
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
                eval_output_path=raw_eval_output_path,
                logger=logger,
            )
            mlflow.set_tag("inference_conducted", "TRUE")
            mlflow.log_metric("number_of_empty_responses", results)
            mlflow.log_artifact(raw_eval_output_path)
            clear_gpu_cache()

        if experiment_config.run_eval:
            print("Running evaluation...")
            prepare_files(
                raw_eval_output_path, eval_output_path, trec_eval_output_path, run_name
            )
            scores = calculate_metrics(
                eval_output_path=eval_output_path,
                gold_labels_file=qrels_2022_path,
                ft_model_name=experiment_config.ft_model,
                run_name=run_name,
                logger=logger,
            )
            mlflow.set_tag("evaluation_conducted", "TRUE")
            mlflow.log_metrics(scores)
            clear_gpu_cache()

        # Clean gpu_cache before next mlflow run
        clear_gpu_cache()
        logger.debug(f"Run with ID {run.info.run_id} finished successful")
    # Set experiment ID
    # Run experiments from experiment.yaml
    # Train model on different parameters
    # Run llama_testing.py with fine-tuned model variation
    # Output file with predicted classes
    # Use files to calculate metrics as accuracy, F1 and AUC, etc.


if __name__ == "__main__":
    fire.Fire(main)
