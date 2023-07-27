import os
import re

import pandas as pd
import argparse
import shutil
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

debug = args.debug


base_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
home_directory = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
data_directory = os.path.join(home_directory, "data")

config_file = os.path.join(base_directory, "configs/config.yaml")
with open(config_file, "r") as file:
    config = yaml.safe_load(file)


def read_qrel_txt(path: str):
    qrels = pd.read_csv(
        os.path.join(home_directory, "data/2021/"),
        sep=" ",
        header=None,
        names=["Topic", "N/A", "Clinical Trial ID", "Label"],
    )

    if debug:
        print(qrels.head(10))
        qrels = qrels.iloc[0]

    return qrels


def copy_files_from_json(qrels: pd.DataFrame, target_dir: str) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in qrels["Clinical Trial ID"]:
        source_dir = search_target_directory(filename)
        full_filename = filename + ".xml"
        source_file = os.path.join(source_dir, full_filename)
        target_file = os.path.join(target_dir, full_filename)

        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            if debug:
                print(f"Copied {filename} to {target_dir}")
        else:
            print(f"File {filename} not found in {source_dir}")


def search_target_directory(filename):
    directories = os.listdir(data_directory)
    for directory in directories:
        for sub_dir in directory:
            pattern = r"^" + re.escape(sub_dir[:7])
            if bool(re.search(pattern, filename[:7])):
                return os.path.join(
                    data_directory, config["year_of_data"], directory, sub_dir
                )


if __name__ == "__main__":
    # Replace these paths with your actual paths
    qrel_path = os.path.join(
        data_directory,
        config["year_of_data"],
        "GoldStandard/trec.nist.gov_data_trials_qrels2021.txt",
    )
    target_directory = os.path.join(base_directory, "data/required_cts")
    qrels = read_qrel_txt()
    copy_files_from_json(qrels, target_directory)
