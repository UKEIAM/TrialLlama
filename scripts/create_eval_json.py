# 1. Iterate trough the model's output and select IDc
# 2. Create new JSON, concatenating the model's outputs with the 'topic' and 'clinical_trial' entry from ct_test_sample_v6.json file

import os
import json
import pandas as pd

base_dir = os.path.dirname(os.path.dirname((__file__)))

model_file_path = os.path.join(
    base_dir, "out", "eval", "eval_bs105o6y-784_v9_4_0.0001_0.1.json"
)
sample_file_path = os.path.join(base_dir, "data", "ct_test_v9.json")
# Load the main data JSON
with open(model_file_path, "r") as json_file:
    model_output = json.load(json_file)

# Load the response data JSON
with open(sample_file_path) as response_file:
    response_data = json.load(response_file)

output = pd.DataFrame(model_output)
output = output.rename(columns={"ID": "id"})

testing_set = pd.DataFrame(response_data)
merged_df = testing_set.merge(output, on=["id"], suffixes=("_model", "_truth"))
# TODO: Double check if first item in response equals first item in sample file (it should)
eval_list = []
for entry in merged_df.iterrows():
    new_entry = {
        "id": entry[1]["id"],
        "ground_truth": entry[1]["output"],
        "topic": entry[1]["topic"],
        "clinical_trial": entry[1]["clinical_trial"],
        "response": entry[1]["RESPONSE"],
        "certainty": entry[1]["PROBA"],
    }
    eval_list.append(new_entry)

df = pd.DataFrame(eval_list)
base_dir = os.path.dirname(os.path.dirname(__file__))
eval_df = df.sample(n=30, random_state=42, ignore_index=True)
df.to_json(
    os.path.join(base_dir, "out", "eval", "summary_eval_file_full.json"),
    orient="records",
)
eval_df.to_json(
    os.path.join(base_dir, "out", "eval", "summary_eval_file_30.json"), orient="records"
)
