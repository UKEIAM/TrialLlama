import os
import re

import pandas as pd
import argparse
import shutil
import yaml

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

debug = args.debug


base_directory = os.path.dirname(os.path.dirname((__file__)))
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_directory = os.path.join(home_directory, "data")

config_file = os.path.join(base_directory, "configs/config.yaml")
with open(config_file, "r") as file:
    config = yaml.safe_load(file)
    
source_data_directory = os.path.join(data_directory, config["year_of_data"])




def read_qrel_txt(qrel_path: str):
    qrels = pd.read_csv(
        qrel_path,
        sep=" ",
        header=None,
        names=["Topic", "N/A", "Clinical Trial ID", "Label"],
    )

    return qrels



"""Function takes a DataFrame as input and searches for all listed Clinical Trial IDs within the data, since most of the trials are not required for the fine-tune dataset"""
def copy_required_files_to_folder(qrels: pd.DataFrame, target_dir: str) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for filename in tqdm(qrels["Clinical Trial ID"]):
        source_dir = search_target_directory(filename)
        full_filename = filename + ".xml"
        source_file = os.path.join(source_dir, full_filename)
        target_file = os.path.join(target_dir)

        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            if debug:
                print(f"Copied {full_filename} to {target_dir}")
        else:
            print(f"File {filename} not found in {source_dir}")
    print("-----FINISHED EXTRACTION-----")


def search_target_directory(filename):
    directories = [dir for dir in os.listdir(os.path.join(source_data_directory)) if os.path.isdir(os.path.join(source_data_directory, dir))]
    for directory in directories:
        sub_dirs = os.listdir(os.path.join(source_data_directory, directory))
        for sub_dir in sub_dirs:
            pattern = re.escape(sub_dir[:7])
            if bool(re.search(pattern, filename[:7])):
                return os.path.join(
                    data_directory, config["year_of_data"], directory, sub_dir
                )


if __name__ == "__main__":
    # Replace these paths with your actual paths
    qrel_path = os.path.join(
        data_directory,
        config["year_of_data"],
        config["qrels_path"],
    )
    target_data_directory = os.path.join(data_directory, "required_cts")
    qrels = read_qrel_txt(qrel_path)
    copy_required_files_to_folder(qrels, target_data_directory)
