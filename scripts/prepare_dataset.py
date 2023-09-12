"""This script will only work with TREC 2021/22 data, since they changed the input format in 2023"""

import os
import re
import json
import sys
import yaml

import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from pathlib import Path
from typing import Optional

# from configs.config import train_config, eval_config

# sys.path.append(str(Path(__file__).resolve().parent.parent))

base_directory = os.path.dirname(os.path.dirname((__file__)))
data_directory = os.path.join(base_directory, "data")
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
raw_ct_data_directory = os.path.join(home_directory, "data")

train_list = []


def create_JSON(
    config_name: Optional[str] = "train",
    out_file_name: Optional[str] = "clinical_trials.json",
    samples: Optional[str] = "all",
    only_criteria: Optional[bool] = False,
):

    if config_name == "train":
        config_file = os.path.join(base_directory, "configs/train_data.yaml")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        config_file = os.path.join(base_directory, "configs/test_data.yaml")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

    source_data_directory = os.path.join(raw_ct_data_directory, config["year_of_data"])
    required_data_directory = os.path.join(
        raw_ct_data_directory, f"{config['mode']}required_cts"
    )

    topics_df = parse_XML_to_df(
        os.path.join(source_data_directory, "topics2021.xml"), ["number", "topic"]
    )
    topics_df["topic"] = topics_df["topic"].replace("\n", " ", regex=True)

    qrel_path = os.path.join(
        raw_ct_data_directory,
        config["year_of_data"],
        config["qrels_path"],
    )
    qrels = read_qrel_txt(qrel_path)
    if samples != "all":
        qrels = qrels[: int(samples)]

    counter = 0
    for index, row in tqdm(qrels.iterrows()):
        topic_nr = row["topic"]
        try:
            topic = topics_df[topics_df["number"] == str(topic_nr)]["topic"].values[0]
            cleaned_topic = clean_textblock(topic)
        except KeyError as e:
            continue
        ct = row["clinical trial id"] + ".xml"
        label = row["label"]
        ct_path = os.path.join(required_data_directory, ct)

        if os.path.exists(ct_path):
            clinical_trial_dict = parse_XML_to_json(ct_path)
            if only_criteria:
                clinical_trial_dict = extract_criteria_from_clinical_trials(
                    clinical_trial_dict
                )
            ct_textblock = extract_textblocks_from_clinical_trials(clinical_trial_dict)
            cleaned_ct_textblocks = []
            for textblock in ct_textblock:
                cleaned_textblock = clean_textblock(textblock)
                cleaned_ct_textblocks.append(cleaned_textblock)
            # TODO: Delete, just for testing reasons
            if label == "0":
                output_text = f"The clinical trial is not relevant for the patient at hand. Status code {label}"
            elif label == "1":
                output_text = f"The patient at hand is not eligible for the clinical presented clinical trial. Status code {label}"
            else:
                output_text = f"The clinical trial fits on the patient's profile. Status code {label}"

            item = {
                "id": f"{index}_{topic_nr}_{ct}",  # ID has following format __index_topicID_ClinicalTrialID__
                "instruction": "Please match the eligibility of following patient to the succeeding clinical trial provided. If the patient profile fits the trial return '2' as answer, which means patient is eligible. If it does not match to the patient profile, return '1' as answer, which means patient is not-eligible. If the trial is not relevant for the patient, return '0' as answer.",
                "input": f"PATIENT DESCRIPTION: {cleaned_topic}\nCLINICAL TRIAL DESCRIPTION: {cleaned_ct_textblocks}",
                "output": str(label),
            }

            full_text_size = item["instruction"] + item["input"]
            if (
                len(full_text_size.split()) > 1000
            ):  # TODO: The current way of creating the dataset concats all available textblock elements within one clinical trial xml. A GPU with 24GB can only handle an max number of input words of 1900. Hence we have to skip all items which are above
                print(f"{ct} nr of words: {len(full_text_size.split())} Skipping...")
                continue
            else:
                train_list.append(item)
                counter += 1
        else:
            continue

    """The below function is returning the full CT parsed into a JSON format + cleaned"""
    # cleaned_list = clean_textblock_data_recursively(train_list)
    out_directory = os.path.join(data_directory, out_file_name)
    print(f"Saving dataset to {out_directory}...")

    with open(out_directory, "w") as fp:
        json.dump(train_list, fp, indent=4)

    print(f"Saved dataset with {counter} examples")


def clean_textblock(text):
    # pattern = r'[^\x00-\x7F]'
    # cleaned_text = re.sub(pattern, "", text)
    cleaned_text = text.replace(r'\\"', r"'")
    # cleaned_text = re.sub(r'[@#$*_{}\[\]"\'\|\\~`]', ' ', cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text.strip())
    return cleaned_text


def clean_textblock_data_recursively(json_obj: list | dict) -> dict:
    """Recursive function to find nested 'textblock' elements and clean the string from unnessecary chars and whitespaces"""
    if isinstance(json_obj, dict):
        cleaned_dict = {}
        for key, value in json_obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                cleaned_dict[key] = clean_textblock_data_recursively(value)
            elif key in ["patient_description", "textblock"]:
                cleaned_dict[key] = clean_textblock(value)
            else:
                cleaned_dict[key] = value
        return cleaned_dict
    elif isinstance(json_obj, list):
        cleaned_list = []
        for item in json_obj:
            cleaned_list.append(clean_textblock_data_recursively(item))
        return cleaned_list
    else:
        return json_obj


def extract_textblocks_from_clinical_trials(clinical_trial: list | dict) -> list:
    """Extracts 'textblock' elements from the 'clinical_trials' section"""
    textblocks = []
    if isinstance(clinical_trial, list):
        result_dict = {}
        for d in clinical_trial:
            result_dict.update(d)
        clinical_trial = result_dict
    for key, value in clinical_trial.items():
        if key == "textblock":
            textblocks.append(value)
        elif isinstance(value, dict):
            textblocks.extend(extract_textblocks_from_clinical_trials(value))
    return textblocks


def extract_criteria_from_clinical_trials(clinical_trial: dict) -> list:
    criteria = []
    for key, value in clinical_trial.items():
        if key == "criteria":
            criteria.append(value)
        elif isinstance(value, dict):
            criteria.extend(extract_criteria_from_clinical_trials(value))
    return criteria


def read_qrel_txt(qrel_path: str) -> pd.DataFrame:
    qrels = pd.read_csv(
        qrel_path,
        sep=" ",
        header=None,
        names=["topic", "N/A", "clinical trial id", "label"],
    )

    return qrels


def parse_XML_to_df(xml_file, df_cols) -> pd.DataFrame:
    """Parse the input XML file and store the result in a pandas
    DataFrame with the given columns.

    The first element of df_cols is supposed to be the identifier
    variable, which is an attribute of each node element in the
    XML data; other features will be parsed from the text content
    of each sub-element.
    """
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()
    rows = []

    for node in xroot:
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]:
            if node is not None and node.text is not None:
                res.append(node.text)
            else:
                res.append(None)
        rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})

    out_df = pd.DataFrame(rows, columns=df_cols)

    return out_df


def parse_XML_to_json(xml_file: str) -> dict:
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()
    dict_data = xml_to_dict(xroot)

    return dict_data


def xml_to_dict(element):
    data = {}
    for child in element:
        if list(child):
            data[child.tag] = xml_to_dict(child)
        else:
            data[child.tag] = child.text
    return data


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(create_JSON)
