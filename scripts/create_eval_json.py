# 1. Iterate trough the model's output and select IDc
# 2. Create new JSON, concatenating the model's outputs with the 'topic' and 'clinical_trial' entry from ct_test_sample_v6.json file

import os
import json
import pandas as pd

base_dir = os.path.dirname(os.path.dirname((__file__)))

model_file_path = os.path.join(
    base_dir, "out", "eval", "eval_bs105o6y-784_v9_4_0.0001_0.1.json"
)
sample_file_path = os.path.join(base_dir, "data", "ct_test_sample_v9.json")
# Load the main data JSON
with open(model_file_path, "r") as json_file:
    model_output = json.load(json_file)

# Load the response data JSON
with open(sample_file_path) as response_file:
    response_data = json.load(response_file)

# TODO: Double check if first item in response equals first item in sample file (it should)
eval_list = []
for idx, entry in enumerate(model_output):
    assert entry["ID"] == response_data[idx]["id"]
    new_entry = {
        "id": entry["ID"],
        "topic": response_data[idx]["topic"],
        "clinical_trial": response_data[idx]["clinical_trial"],
        "response": entry["RESPONSE"],
    }
    eval_list.append(entry)

df = pd.DataFrame(eval_list)
base_dir = os.path.dirname(os.path.dirname(__file__))
df.to_json(
    os.path.join(base_dir, "out", "eval", "summary_eval_file.json"), orient="records"
)
