import json

data_path = "data/ct_21.json"
data_path_2 = "data/ct_22.json"

with open(data_path, 'r') as file:
    data = json.load(file)

with open(data_path_2, 'r') as file:
    data_2 = json.load(file)

out_directory = "data/ct_all_years_v3.json"
merged = data + data_2

with open(out_directory, "w") as fp:
    json.dump(merged, fp, indent=4)
