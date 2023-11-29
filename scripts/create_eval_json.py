# 1. Iterate trough the model's output and select IDc
# 2. Create new JSON, concatenating the model's outputs with the 'topic' and 'clinical_trial' entry from ct_test_sample_v6.json file

import os
import json
import re
import pandas as pd

base_dir = os.path.dirname(os.path.dirname((__file__)))
run_name = "lbn5d6tp-748_v9_50_4_v3"
model_file_path = os.path.join(base_dir, "out", "eval", f"eval_{run_name}.json")
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
summary_list = []
response_list = []
for entry in merged_df.iterrows():
    new_entry_sum = {
        "ID": entry[1]["id"],
        "Ground Truth": entry[1]["output"],
        "Topic": entry[1]["topic"],
        "Clinical Trial": entry[1]["clinical_trial"],
        "Model Certainty": entry[1]["PROBA"],
        "Helper": "Likert-Scale Helper: (1) Very poor, (2) Poor, (3) Fair, (4) Good, (5) Excellent",
    }
    summary_list.append(new_entry_sum)

    new_entry_resp = {
        "ID": entry[1]["id"],
        "Model Response": entry[1]["RESPONSE"],
    }
    response_list.append(new_entry_resp)

df_sum = pd.DataFrame(summary_list)
df_resp = pd.DataFrame(response_list)
base_dir = os.path.dirname(os.path.dirname(__file__))
eval_sum = df_sum.sample(n=20, random_state=42, ignore_index=True)
eval_resp = df_resp.sample(n=20, random_state=42, ignore_index=True)
eval_sum.to_json(
    os.path.join(base_dir, "out", "eval", f"summary_eval_file_{run_name}_sum.json"),
    orient="records",
)
eval_resp.to_json(
    os.path.join(base_dir, "out", "eval", f"summary_eval_file_{run_name}_resp.json"),
    orient="records",
)


# Second part ---------------------------------------------------------------------------------------------------------

eval_sum_json = (
    os.path.join(base_dir, "out", "eval", f"summary_eval_file_{run_name}_sum.json"),
)

eval_resp_json = (
    os.path.join(base_dir, "out", "eval", f"summary_eval_file_{run_name}_resp.json"),
)

# Load the JSON data from your JSON file
with open(eval_sum_json[0], "r") as json_file:
    data = json.load(json_file)

dir_path = os.path.join(base_dir, "out", "eval", "qual_eval")
txt_path = os.path.join(base_dir, dir_path, f"eval_{run_name}_sum.html")
os.makedirs(dir_path, exist_ok=True)
# Open a text file for writing
with open(txt_path, "w") as text_file:
    for item in data:
        for key, value in item.items():
            # Write the capitalized key in bold
            if key in ["ID", "Ground Truth", "Model Certainty"]:
                text_file.write(f"<strong>{key}</strong> ")
            else:
                text_file.write(f"<strong>{key}</strong><br> ")

            text_file.write(f"{value}<br>")


with open(eval_resp_json[0], "r") as json_file:
    data = json.load(json_file)

dir_path = os.path.join(base_dir, "out", "eval", "qual_eval")
txt_path_resp = os.path.join(base_dir, dir_path, f"eval_{run_name}_resp.html")
os.makedirs(dir_path, exist_ok=True)
# Open a text file for writing
with open(txt_path_resp, "w") as text_file:
    for item in data:
        for key, value in item.items():
            # Write the capitalized key in bold
            if key in ["Model Response"]:
                text_file.write(f"<br><strong>{key}</strong> ")

            if key == "Model Response":
                value = re.sub(r"#\d+", r"<br>\g<0>", value)
                text_file.write(f"{value}<br><br><br>")

print("FINISHED")
