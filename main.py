# TODO: Prepare experiment script
import os
import logging
import fire
import random
import string

import mlflow
import mlflow.pytorch

from utils.train_utils import clear_gpu_cache
from finetuning import main as ft_main
from inference_main import main as test_main
from utils.eval_utils import (
    calculate_metrics,
    prepare_files,
    prepare_binary,
    evaluate_binary,
)
from configs.experiments import experiment_config
from utils.config_utils import update_config
from utils.logger_utils import setup_logger
from utils.set_free_cuda_device import get_free_cuda_device

base_dir = os.path.dirname((__file__))


def main(**kwargs):
    update_config((experiment_config), **kwargs)
    if "dataset_name" in kwargs:
        experiment_config.dataset_name = kwargs["dataset_name"]
    else:
        experiment_config.dataset_name = (
            f"ct_train_sample_{experiment_config.dataset_version}"
        )
    qrels_dir = os.path.join(base_dir, "data", "gold_labels")
    experiment_config_dir = os.path.join(
        base_dir, "configs", "experiment_definitions.yaml"
    )
    train_plt_path = plt_save_path = os.path.join(
        "out", "eval", "img", f"{experiment_config.ft_model}_loss_vs_epoch.png"
    )
    if experiment_config.evaluate_base_model:
        experiment_name = (
            f"{experiment_config.base_model.lower()}-base-{experiment_config.task}"
        )
    else:
        experiment_name = f"{experiment_config.base_model.lower()}-{experiment_config.dataset_version}-{experiment_config.dataset_size}-{experiment_config.task}"
    mlflow.set_experiment(experiment_name)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    mlflow.set_tracking_uri(os.path.join(base_dir, "mlruns"))
    description = (
        f"Fine-tuned model {experiment_config.ft_model} | Dataset balancing v3"
    )
    # Define the characters to choose from for the prefix
    prefix_characters = string.ascii_lowercase + string.digits
    prefix_length = 8  # Adjust the length as needed
    # Generate a random prefix
    prefix = "".join(random.choice(prefix_characters) for _ in range(prefix_length))
    # Define the length of the random number
    number_length = 3
    # Generate a random number with the specified length
    random_number = "".join(random.choice(string.digits) for _ in range(number_length))
    # Combine the random prefix and random number to create the run_name
    rand_name = f"{prefix}-{random_number}"
    run_name = f"{rand_name}_{experiment_config.dataset_test_version}_{experiment_config.batch_size}_{experiment_config.lr}_{experiment_config.temperature}"
    with mlflow.start_run(description=description, run_name=run_name) as run:
        logger = setup_logger(run_id=run.info.run_id, run_name=run_name)
        eval_output_path = os.path.join(
            base_dir,
            "out",
            "eval",
            f"eval_{run_name}.json",
        )
        mlflow.log_params(
            {
                "base_model": experiment_config.base_model,
                "batch_size": experiment_config.batch_size,
                "num_epochs": experiment_config.num_epochs,
                "learning_rate": experiment_config.lr,
                "weight_decay": experiment_config.weight_decay,
                "dataset_name": experiment_config.dataset_name,
                "dataset_version": experiment_config.dataset_version,
                "dataset_size_testing": experiment_config.dataset_size_testing,
                "one_shot_example": experiment_config.add_example,
                "dataset_test_version": experiment_config.dataset_test_version,
                "dataset_size": experiment_config.dataset_size,
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
                dataset=experiment_config.dataset_name,
                dataset_size=experiment_config.dataset_size,
                create_sample=experiment_config.create_sample,
                lr=experiment_config.lr,
                num_epochs=experiment_config.num_epochs,
                model_name=experiment_config.base_model,
                ft_model=experiment_config.ft_model,
                max_tokens=experiment_config.max_tokens,
            )
            mlflow.set_tag("ft_conducted", "TRUE")
            mlflow.log_metrics(results)
            # mlflow.log_artifact(local_path=train_plt_path)
            clear_gpu_cache()

        if experiment_config.run_inference:
            print("Running inference...")
            results = test_main(
                dataset_size=experiment_config.dataset_size_testing,
                dataset_version=experiment_config.dataset_test_version,
                dataset=f"ct_test_sample_{experiment_config.dataset_test_version}",
                model_name=experiment_config.base_model,
                ft_model=experiment_config.ft_model,
                load_peft_model=experiment_config.load_peft_model,
                max_new_tokens=experiment_config.max_new_tokens,
                temperature=experiment_config.temperature,
                top_k=experiment_config.top_k,
                top_p=experiment_config.top_p,
                length_penalty=experiment_config.length_penalty,
                repetition_penalty=experiment_config.repetition_penalty,
                debug=experiment_config.debug,
                eval_output_path=eval_output_path,
                logger=logger,
                evaluate_base_model=experiment_config.evaluate_base_model,
                add_example=experiment_config.add_example,
            )
            mlflow.set_tag("inference_conducted", "TRUE")
            mlflow.log_metric("number_of_empty_responses", results)
            # artifact_uri = os.path.join(
            #     base_dir, "mlruns", run.info.experiment_id, run.info.run_id, "artifacts"
            # )
            try:
                # TODO: Suddenly some issues with logging artifacts happened because of PermissionError on remote interpreter"
                mlflow.log_artifact(local_path=experiment_config_dir)
                mlflow.log_artifact(local_path=eval_output_path)
            except Exception as e:
                logger.error(f"Error while logging artifact: {e}")
            clear_gpu_cache()

        if experiment_config.run_eval:
            try:
                print("Running evaluation...")
                eval_df = prepare_files(
                    eval_output_path=eval_output_path, run_name=run_name, logger=logger
                )
                scores = calculate_metrics(
                    eval_df=eval_df,
                    gold_labels_dir=qrels_dir,
                    ft_model_name=experiment_config.ft_model,
                    run_name=run_name,
                    logger=logger,
                )

                binary_val_df = prepare_binary(
                    eval_output_path=eval_output_path, run_name=run_name, logger=logger
                )
                binary_scores = evaluate_binary(
                    eval_df=binary_val_df,
                    gold_labels_dir=qrels_dir,
                    ft_model_name=experiment_config.ft_model,
                    run_name=run_name,
                    logger=logger,
                )
                mlflow.set_tag("evaluation_conducted", "TRUE")
                mlflow.log_metrics(scores)
                mlflow.log_metrics(binary_scores)
                clear_gpu_cache()

            except Exception as e:
                logger.error(f"Output does not corresponds to the required format: {e}")

        # Clean gpu_cache before next mlflow run
        print("Run completed successfully")
        logger.debug(f"Run with ID {run.info.run_id} finished successful")
        clear_gpu_cache()
    # Set experiment ID
    # Run experiments from experiment.yaml
    # Train model on different parameters
    # Run llama_testing.py with fine-tuned model variation
    # Output file with predicted classes
    # Use files to calculate metrics as accuracy, F1 and AUC, etc.


if __name__ == "__main__":
    fire.Fire(main)
