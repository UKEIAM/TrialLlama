"""This script will only work with TREC 2021/22 data, since they changed the input format in 2023"""

import os
import re
import json
import sys
import yaml
import torch

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

data_list = []


def create_JSON(
    config_name: Optional[str] = "train",
    out_file_name: Optional[str] = "ct_full_v3.json",
):

    if config_name == "train":
        config_file = os.path.join(base_directory, "configs/train_data.yaml")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        config_file = os.path.join(base_directory, "configs/test_data.yaml")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

    source_data_directory = os.path.join(
        raw_ct_data_directory, str(config["year_of_data"])
    )
    required_data_directory = os.path.join(
        raw_ct_data_directory, f"{config['mode']}required_cts"
    )

    topics_df = parse_XML_to_df(
        os.path.join(source_data_directory, f"topics{config['year_of_topics']}.xml"),
        ["number", "topic"],
    )
    topics_df["topic"] = topics_df["topic"].replace("\n", " ", regex=True)

    qrel_path = os.path.join(
        raw_ct_data_directory,
        str(config["year_of_data"]),
        config["qrels_path"],
    )
    qrels = read_qrel_txt(qrel_path)

    # DEBUG
    # qrels = qrels[:10]

    counter = 0
    for index, row in tqdm(qrels.iterrows()):
        topic_nr = row["topic"]
        try:
            topic = topics_df[topics_df["number"] == str(topic_nr)]["topic"].values[0]
            cleaned_topic = clean_textblock(topic)
        except KeyError as e:
            continue
        ct = row["clinical trial id"]
        label = row["label"]
        ct_path = os.path.join(required_data_directory, ct + ".xml")

        if os.path.exists(ct_path):
            clinical_trial_dict = parse_XML_to_json(ct_path)
            ct_data = extract_required_data_from_clinical_trials(clinical_trial_dict)
            if label == 0:
                category = "no relevant information"
                output_text = f"The clinical trial is not relevant for the patient at hand. Status code {label}"
            elif label == 1:
                category = "not eligible"
                output_text = f"The patient at hand is not eligible for the clinical presented clinical trial. Status code {label}"
            else:
                category = "eligible"
            output_text = (
                f"The clinical trial fits on the patient's profile. Status code {label}"
            )
            item = {
                "id": f"{index}_{topic_nr}_{ct}",
                "instruction": "Hello. You are a helpful assistant for clinical trial recruitment."
                "Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility. "
                "The factors that allow someone to participate in a clinical study are called inclusion criteria. "
                "They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions. "
                "You should check the inclusion and exclusion criteria one-by-one. If at least one exclusion criterion is met, the patient is automaticall not eligible."
                "For each inclusion criterion, first think step-by-step to explain if and how the patient note is relevant to the criterion."
                "Your answer should be in the following format: dict{str(inclusion_criterion): list[str(relevance_explanation)} str('eligible'|'not eligible'|'no relevant information')]}\n\n",
                "input": f"Here is the example patient note: {cleaned_topic}\n\nHere is the clinical trial: {ct_data}",
                "output": category,
            }
            # item = {
            #     "id": f"{index}_{topic_nr}_{ct}",
            #     "instruction": "Hello. You are a helpful assistant for clinical trial recruitment."
            #     "Your task is to compare a given patient note and the inclusion criteria of a clinical trial to determine the patient's eligibility. "
            #     "The factors that allow someone to participate in a clinical study are called inclusion criteria. "
            #     "They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions. "
            #     "You should check the inclusion and exclusion criteria one-by-one. "
            #     "You should check the inclusion criteria one-by-one, following the steps below: "
            #     "1. For each inclusion criterion, first think step-by-step to explain if and how the patient note is relevant to the criterion. You should explain in detail if there is relevant information. "
            #     "2. Then, if there is relevant information, you must annotate a list of relevant sentence IDs of the patient note. If there is no relevant information, you must annotate an empty list. "
            #     "3. Finally, annotate the patient eligibility for this specific inclusion criterion: the eligibility must be 'no relevant information' if there is no relevant information. Otherwise, the patient can only be 'included' or 'not included' if there are relevant sentences. 'included' means that the patient meets the inclusion criterion, and 'not included' means that the patient contradicts the inclusion criterion. "
            #     "4. You should output only a JSON dict exactly formatted as: dict{str(inclusion_criterion): list[str(relevance_explanation), list[int(sentence_id)], str('included'|'not included'|'no relevant information')]}\n\n",
            #     "input": f"Here is the example patient note: {cleaned_topic}\n\nHere is the clinical trial: {ct_data}",
            #     "output": category,
            # }

            data_list.append(item)
            counter += 1
        else:
            continue

    """The below function is returning the full CT parsed into a JSON format + cleaned"""
    out_directory = os.path.join(data_directory, out_file_name)
    print(f"Saving dataset to {out_directory}...")

    with open(out_directory, "w") as fp:
        json.dump(data_list, fp, indent=4)

    print(f"Saved dataset with {counter} examples")


def clean_textblock(text):
    cleaned_text = text.replace(r'\\"', r"'")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text.strip())
    return cleaned_text


def extract_data_info(element_type, elements):
    info = []
    for key, value in elements.items():
        if element_type == "eligibility":
            if key == "criteria":
                if value is not None:
                    textblock_element = value.get("textblock")
                    criteria_type = None
                    criteria_dict = {"Inclusion Criteria": [], "Exclusion Criteria": []}
                    if textblock_element.startswith("Inclusion Criteria:"):
                        criteria_type = "Inclusion Criteria"
                    elif textblock_element.startswith("Exclusion Criteria:"):
                        criteria_type = "Exclusion Criteria"
                    else:
                        # Extract bullet-pointed criteria
                        criteria = re.findall(
                            r"\d+\.\s(.*?)(?=(?:\d+\.\s)|$)", textblock_element
                        )
                        criteria_dict[criteria_type].extend(criteria)
                    value = clean_textblock(textblock_element)
            info.append(f"{key.capitalize()}: {value},\n")
        if element_type == "brief_summary":
            value = clean_textblock(value)
            info.append(f"{value}\n")
    return "".join(info)


def extract_required_data_from_clinical_trials(clinical_trial: list | dict) -> list:
    """Extracts all eligibility information from the 'clinical_trials' section"""
    elements = []

    if isinstance(clinical_trial, list):
        result_dict = {}
        for d in clinical_trial:
            result_dict.update(d)
        clinical_trial = result_dict

    for key, value in clinical_trial.items():
        if key == "eligibility" and isinstance(value, dict):
            key = "Eligibility Criteria"
            info = extract_data_info(key, value)
            elements.append(info)
        if key == "brief_summary" and isinstance(value, dict):
            key = "Summary"
            info = extract_data_info(key, value)
            elements.append(info)
        if key == "brief_title":
            key = "Title"
            elements.append(f"{key.capitalize()}: {value}\n")
        elif isinstance(value, (list, dict)):
            elements.extend(extract_required_data_from_clinical_trials(value))

    return elements


# def extract_criteria_from_clinical_trials(clinical_trial: dict) -> list:
#     criteria = []
#     for key, value in clinical_trial.items():
#         if key == "criteria":
#             criteria.append(value)
#         elif isinstance(value, dict):
#             criteria.extend(extract_criteria_from_clinical_trials(value))
#     return criteria


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
