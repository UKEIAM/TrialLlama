import os
import re
import json
import yaml

import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm


base_directory = os.path.dirname(os.path.dirname((__file__)))
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_directory = os.path.join(home_directory, "data")

config_file = os.path.join(base_directory, "configs/config.yaml")
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

source_data_directory = os.path.join(data_directory, config["year_of_data"])
target_data_directory = os.path.join(data_directory, "required_cts")

train_dict = []


def create_JSON(qrles):
    topics_df = parse_XML_to_df(
        os.path.join(source_data_directory, "topics2021.xml"), ["number", "topic"]
    )
    topics_df["topic"] = topics_df["topic"].replace("\n", " ", regex=True)

    for index, row in tqdm(qrles.iterrows()):
        topic_nr = row["topic"]
        try:
            topic = topics_df[topics_df["number"] == str(topic_nr)]["topic"].values[0]
        except KeyError as e:
            continue
        ct = row["clinical trial id"] + ".xml"
        label = row["label"]
        ct_path = os.path.join(target_data_directory, ct)

        if os.path.exists(ct_path):
            """Currently it's simply a JSON file"""
            # clinical_trial_df = parse_XML_to_df(ct_path, ['brief_title', 'officiel_title', 'brief_summary', 'start_date' 'overall_status', 'study_pop', 'sampling_method', 'criteria', 'gender', 'minimum_age', 'maximum_age', 'healthy_volunteers'])
            clinical_trial_dict = parse_XML_to_json(
                ct_path,
                [
                    "brief_title",
                    "officiel_title",
                    "brief_summary",
                    "start_date" "overall_status",
                    "study_pop",
                    "sampling_method",
                    "criteria",
                    "gender",
                    "minimum_age",
                    "maximum_age",
                    "healthy_volunteers",
                ],
            )
            train_dict.append(
                {
                    "id": f"{index}_{topic_nr}_{ct}",  # ID has following format __index_topicID_ClinicalTrialID__
                    "instruction": "Please match the eligibility of following patient to the succeeding clinical trial provided.",
                    "inputs": [
                        {"patient_description": f"{topic}"},
                        {"clinical_trial": clinical_trial_dict},
                    ],
                    "output": f"{label}",
                }
            )
        else:
            continue
        
    cleaned_dict = clean_textblock_data_recursively(train_dict)

    with open(os.path.join(base_directory, "data", "train_json_full_36000.json"), "w") as fp:
        json.dump(cleaned_dict, fp)

def clean_textblock(text):
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    return cleaned_text

def clean_textblock_data_recursively(json_obj):
    """Recursive function to find nested 'textblock' elements and clean the string from unnessecary chars and whitespaces"""
    if isinstance(json_obj, dict):
        cleaned_dict = {}
        for key, value in json_obj.items():
            if isinstance(value, dict) or isinstance(value, list):
                cleaned_dict[key] = clean_textblock_data_recursively(value)
            elif (key == "textblock" or key == "patient_description"):
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
    
def read_qrel_txt(qrel_path: str):
    qrels = pd.read_csv(
        qrel_path,
        sep=" ",
        header=None,
        names=["topic", "N/A", "clinical trial id", "label"],
    )

    return qrels


def parse_XML_to_df(xml_file, df_cols):
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


def parse_XML_to_json(xml_file, cols):
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
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


def read_clinical_trial():
    pass


def read_patient_topic():
    pass


if __name__ == "__main__":
    qrel_path = os.path.join(
        data_directory,
        config["year_of_data"],
        config["qrels_path"],
    )
    qrels = read_qrel_txt(qrel_path)
    train_json = create_JSON(qrels)

    print("FINISH")