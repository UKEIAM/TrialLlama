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
from sklearn.utils import shuffle

# from configs.config import train_config, eval_config

# sys.path.append(str(Path(__file__).resolve().parent.parent))

base_directory = os.path.dirname(os.path.dirname((__file__)))
data_directory = os.path.join(base_directory, "data")
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
raw_ct_data_directory = os.path.join(home_directory, "data")


def create_JSON(
    versions=None,
):
    if versions is None:
        versions = ["v12"]
    for ver in versions:
        version = ver
        data_list = []

        config_file = os.path.join(base_directory, "configs/ct_data.yaml")
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        counter = 0
        for topic_year in config["year_of_topics"]:
            source_data_directory = os.path.join(
                raw_ct_data_directory, str(config["year_of_data"])
            )
            required_data_directory = os.path.join(
                raw_ct_data_directory, "required_cts"
            )
            os.makedirs(required_data_directory, exist_ok=True)

            topics_df = parse_XML_to_df(
                os.path.join(source_data_directory, f"topics{topic_year}.xml"),
                ["number", "topic"],
            )
            topics_df["topic"] = topics_df["topic"].replace("\n", " ", regex=True)

            qrel_path = os.path.join(
                raw_ct_data_directory,
                str(config["year_of_data"]),
                f"{config['qrels_path']}{topic_year}.txt",
            )
            qrels = read_qrel_txt(qrel_path)

            # DEBUG
            # qrels = qrels[:100]

            for index, row in tqdm(qrels.iterrows()):
                topic_nr = f"{row['topic']}-{topic_year}"
                try:
                    topic = topics_df[
                        topics_df["number"] == str(topic_nr.split("-")[0])
                    ]["topic"].values[0]
                    cleaned_topic = clean_textblock(topic)
                except KeyError as e:
                    continue
                ct = row["clinical trial id"]
                label = row["label"]
                ct_path = os.path.join(required_data_directory, ct + ".xml")
                if os.path.exists(ct_path):
                    clinical_trial_dict = parse_XML_to_json(ct_path)
                    ct_data = extract_required_data_from_clinical_trials(
                        clinical_trial_dict
                    )
                    filter = " ".join(ct_data)
                    if (
                        "inclusion criteria:" in filter.lower()
                        and "exclusion criteria:" in filter.lower()
                    ):
                        ct_input = ct_data.copy()
                        if ct == "NCT00424346":
                            print()
                        for idx, item in enumerate(ct_data):
                            if "eligibility:" in item.lower():
                                index_exclusion = item.lower().find(
                                    "exclusion criteria"
                                )
                                index_inclusion = item.lower().find("inclusion")
                                inclusion_crit = item[index_inclusion:index_exclusion]
                                exclusion_crit = item[index_exclusion:]
                                pattern = (
                                    r"(Inclusion|Exclusion)(\s+Criteria|\s+criteria):?"
                                )
                                replace_inc = "INCLUSION CRITERIA:"
                                replace_exc = "EXCLUSION CRITERIA:"
                                inclusion = re.sub(pattern, replace_inc, inclusion_crit)
                                exclusion = re.sub(pattern, replace_exc, exclusion_crit)
                                inclusion = create_numbered_list(inclusion)
                                exclusion = create_numbered_list(exclusion)
                                # general_inclusion_crit = item[index_gender:]
                                ct_input.pop(idx)
                                # ct_input.insert(idx, f"OVERVIEW: {general_inclusion_crit}\n{inclusion_crit}\n{exclusion_crit}")
                                ct_input.insert(idx, f"{inclusion}\n{exclusion}")
                            else:
                                continue
                        ct_final_input = "\n".join([f"{item}" for item in ct_input])
                    else:
                        continue
                    if label == 0:
                        category = "C: irrelevant"
                    elif label == 1:
                        category = "B: excluded"
                    else:
                        category = "A: eligible"
                    id_string = f"{index}_{topic_nr}_{ct}"
                    version = version.split("_")[0]
                    item = {
                        "id": id_string,
                        "topic_year": topic_year,
                        "instruction": config[version]["instruction"],
                        "topic": f"Here is the patient note:\n{cleaned_topic}",
                        "clinical_trial": f"Here is the clinical trial:\n{ct_final_input}",
                        "response": config[version]["response"],
                        "output": f"{category}",
                    }
                    try:
                        pattern_final_check = (
                            r"(INCLUSION CRITERIA:|EXCLUSION CRITERIA:)"
                        )
                        matches = re.findall(
                            pattern_final_check, item["clinical_trial"]
                        )
                        assert len(matches) == 2
                    except AssertionError as e:
                        continue
                    data_list.append(item)
                    counter += 1
                else:
                    continue

        """The below function is returning the full CT parsed into a JSON format + cleaned"""
        out_directory_train = os.path.join(data_directory, f"ct_train_{version}.json")
        out_directory_test = os.path.join(data_directory, f"ct_test_{version}.json")

        df = pd.DataFrame(data_list)
        # Step 1: Mix the samples
        df = shuffle(df, random_state=42)  # Shuffle the rows randomly

        # Step 2: Create a test dataset which withholds 10 topics from the training dataset
        df["topic_id"] = df["id"].str.split("_").str[1]
        topics_to_filter = [
            "12-2021",
            "15-2021",
            "17-2022",
            "27-2022",
            "30-2021",
            "33-2022",
            "36-2021",
            "40-2022",
            "53-2021",
            "65-2021",
            "71-2021",
            "75-2021",
        ]
        mask = df["topic_id"].isin(topics_to_filter)

        # test_dataset = df.sample(
        #     n=test_samples, random_state=42
        # )  # Randomly select 1000 samples for testing

        # Step 3: Remove the test dataset from the original DataFrame
        train_dataset = df[~mask].copy()
        test_dataset = df[mask].copy()
        train_dataset.drop(["topic_id"], axis=1, inplace=True)
        test_dataset.drop(["topic_id"], axis=1, inplace=True)

        train_dataset.to_json(out_directory_train, orient="records")
        test_dataset.to_json(out_directory_test, orient="records")

        print(f"Saved dataset Version {version} with {counter} examples")


def clean_textblock(text):
    cleaned_text = text.replace(r'\\"', r"'")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text.strip())
    return cleaned_text


def format_criteria(criteria_text):
    return [criterion.strip() for criterion in criteria_text.strip().split("\n")]


def extract_data_info(element_type, elements):
    info = []
    for key, value in elements.items():
        if element_type == "eligibility":
            if key == "criteria":
                textblock_element = value.get("textblock")
                value = clean_textblock(textblock_element)
                info.append(f"{value}")
            else:
                continue
        if element_type == "Summary":
            value = clean_textblock(value)
            info.append(f"{value}")
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
            info = extract_data_info(key, value)
            key = "ELIGIBILITY"
            elements.append(f"{key}: {info}")
        elif key == "brief_summary" and isinstance(value, dict):
            key = "Summary"
            info = extract_data_info(key, value)
            # continue
            elements.append(f"{key}: {info}")
        elif key == "brief_title":
            key = "Title"
            # continue
            elements.append(f"{key}: {value}")
        elif key == "intervention_type":
            key = "Intervention Type"
            # continue
            elements.append(f"{key}: {value}")
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


def create_numbered_list(text: str) -> str:
    splitted = text.split(" - ")
    if len(splitted) == 1:
        return None
    pattern = r"(INCLUSION CRITERIA:|EXCLUSION CRITERIA:)"
    result = ""
    for idx, item in enumerate(splitted):
        if idx != 0:
            result += f"{idx}. {item}\n"
        else:
            text = re.match(pattern, item)
            try:
                assert text != None
            except AssertionError as e:
                print(e)
                return None
            result += f"{text[0]}\n"
    return result


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(create_JSON)
