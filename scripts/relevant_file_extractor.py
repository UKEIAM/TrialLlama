import os
import re

import pandas as pd
import argparse
import shutil
import yaml

from tqdm import tqdm
from configs.config import train_config, eval_config


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

debug = args.debug


base_directory = os.path.dirname(os.path.dirname((__file__)))
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_directory = os.path.join(home_directory, "data")




def read_qrel_txt(qrel_path: str):
    qrels = pd.read_csv(
        qrel_path,
        sep=" ",
        header=None,
        names=["topic", "N/A", "clinical trial id", "label"],
    )

    return qrels


"""Function takes a DataFrame as input and searches for all listed Clinical Trial IDs within the data, since most of the trials are not required for the fine-tune dataset"""


def main(config_name: str = "config") -> None:

    if config_name is "config":
        config = train_config
    else:
        config = eval_config

    source_data_dir = os.path.join(data_directory, config.year_of_data)

    qrel_path = os.path.join(
        data_directory,
        config.year_of_data,
        config.qrels_path,
    )

    target_dir = os.path.join(
        data_directory, f"{config.mode}required_cts"
    )

    qrels = read_qrel_txt(qrel_path)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in tqdm(qrels["clinical trial id"]):
        source_dir = search_target_directory(filename, source_data_dir, config.year_of_data)
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


def search_target_directory(filename, source_data_dir, year_of_data):
    directories = [
        dir
        for dir in os.listdir(os.path.join(source_data_dir))
        if os.path.isdir(os.path.join(source_data_dir, dir))
    ]
    for directory in directories:
        sub_dirs = os.listdir(os.path.join(source_data_dir, directory))
        for sub_dir in sub_dirs:
            pattern = re.escape(sub_dir[:7])
            if bool(re.search(pattern, filename[:7])):
                return os.path.join(
                    data_directory, year_of_data, directory, sub_dir
                )


if __name__ == "__main__":
    from fire import Fire
  
    Fire(main)
