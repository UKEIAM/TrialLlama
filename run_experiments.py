# TODO: Prepare experiment script

import mlflow
import yaml
import os

base_directory = os.path.dirname(os.path.dirname((__file__)))

with open(os.path.join(base_directory, "configs", "experiments.yaml"), "r") as file:
  experiment_config = yaml.safe_load(file)

for item in experiment_config:
  with mlflow as run:
    # Set experiment ID
    # Run experiments from experiment.yaml
      # Train model on different parameters
      # Run llama_testing.py with fine-tuned model variation
      # Output file with predicted classes
      # Use files to calculate metrics as accuracy, F1 and AUC, etc.
    pass