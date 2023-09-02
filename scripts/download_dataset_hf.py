import os
import json

from datasets import load_dataset


base_directory = os.path.dirname(os.path.dirname((__file__)))

# Define the dataset name and version
dataset_name = "tatsu-lab/alpaca"

# Load the dataset
dataset = load_dataset(dataset_name)
split_name = "train"

data = dataset[split_name].to_pandas().to_dict(orient="records")

# Save the dataset as a JSON file
output_file = os.path.join(base_directory, "data", "alpaca_data.json")

with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Dataset saved as {output_file}")