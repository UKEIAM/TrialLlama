import os

from utils.eval_utils import calculate_metrics
from configs.experiments import experiment_config

eval_output_path = os.path.join("out", "eval", f"eval_llama-13b-300_qrels.txt")

qrels_2022_path = os.path.join(
    "data",
    f"trec.nist.gov_data_trials_qrels2022.txt",
)


scores = calculate_metrics(
    eval_output_path=eval_output_path,
    gold_labels_file=qrels_2022_path,
    ft_model_name=experiment_config.ft_model,
)


print(scores)
