import os
import re
import json
import yaml

import pandas as pd
import xml.etree.ElementTree as ET



base_directory = os.path.dirname(os.path.dirname((__file__)))
home_directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_directory = os.path.join(home_directory, 'data')

config_file = os.path.join(base_directory, 'configs/config.yaml')
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
    
source_data_directory = os.path.join(data_directory, config['year_of_data'])
target_data_directory = os.path.join(data_directory, 'required_cts')

clinical_trials_json = []



def create_JSON(qrles):
    topics_df = parse_XML(os.path.join(source_data_directory, 'topics2021.xml'), ['number', 'topic'])
    topics_df['topic'] = topics_df['topic'].replace('\n', ' ', regex=True)

    for index, row in qrles.iterrows():
        topic = row['topic']
        ct = row['clinical trial id']+'.xml'
        label = row['label']
        ct_path = os.path.join(target_data_directory, ct)

        """Currently it's simply a JSON file"""
        clinical_trial_df = parse_ct_XML(ct_path, ['brief_title', 'officiel_title', 'brief_summary', 'start_date' 'overall_status', 'study_pop', 'sampling_method', 'criteria', 'gender', 'minimum_age', 'maximum_age', 'healthy_volunteers'])

        clinical_trials_json.append(
            {
                "id": f'{index}{row["topic"]}{row["clinical trial id"]}',
                "instruction": "Please match the eligibility of following patient to the succeeding clinical trial provided.",
                "inputs": [ {
                    "patient_description": "" },
                {
                    "clinical_trial": "YOUR CLINICAL TRIAL"
                }
                ],
                "output": "0: non-relevant/ 1: excluded/ 2: eligible"
            }
        )



def read_qrel_txt(qrel_path: str):
    qrels = pd.read_csv(
        qrel_path,
        sep=' ',
        header=None,
        names=['topic', 'N/A', 'clinical trial id', 'label'],
    )

    return qrels

def parse_XML(xml_file, df_cols): 
    '''Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    '''
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
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df

def parse_ct_XML(xml_file, cols):
    xtree = ET.parse(xml_file)
    xroot = xtree.getroot()
    rows = []

    json_data = xml_to_dict(xroot)
    json_string = json.dumps(json_data, indent=2)

    """TODO: Checkout if simply parsing whole XML as JSON and inputting it into the model works. Otherwise, XML needs to be extracted accordingly."""

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



if __name__ == '__main__':
    qrel_path = os.path.join(
        data_directory,
        config['year_of_data'],
        config['qrels_path'],
    )
    qrels = read_qrel_txt(qrel_path)
    create_JSON(qrels)
