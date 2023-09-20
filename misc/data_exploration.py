import os

import pandas as pd

base_dir = os.path.dirname(os.path.dirname(__file__))

type = "train"

path = base_dir
out_path = base_dir
if type == "train":
    path = os.path.join(base_dir, "data", "ct_full.json")
    out_path = os.path.join(base_dir, "data", "ct.json")
elif type == "test":
    path = os.path.join(base_dir, "data", "ct_testing_full.json")
    out_path = os.path.join(base_dir, "data", "ct_testing.json")

df = pd.read_json(path)
