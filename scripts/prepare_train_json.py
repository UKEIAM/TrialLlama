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




def create_JSON(qrles):
    topics_df = parse_XML(os.path.join(source_data_directory, 'topics2021.xml'), ['number', 'topic'])
    topics_df['topic'] = topics_df['topic'].replace('\n', ' ', regex=True)
    for index, row in qrles.iterrows():
        pass


def read_qrel_txt(qrel_path: str):
    qrels = pd.read_csv(
        qrel_path,
        sep=' ',
        header=None,
        names=['Topic', 'N/A', 'Clinical Trial ID', 'Label'],
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
