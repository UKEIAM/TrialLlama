from dataclasses import dataclass

@dataclass
class train_config:
  mode: str = ""
  year_of_data: str = "2021"
  year_of_topics: str = "2021"
  qrels_path: str = "GoldStandard/trec.nist.gov_data_trials_qrels2021.txt"

@dataclass
class eval_config:
  mode: str = "eval_"
  year_of_data: str = "2021"
  year_of_topics: str =  "2022"
  qrels_path: str = "GoldStandard/trec.nist.gov_data_trials_qrels2022.txt"