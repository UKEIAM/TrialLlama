import os

from utils.eval_utils import prepare_files, calculate_metrics

raw_eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_llama-2-13b-chat-hf-300-v2-4-one_languid-stoat-38_v3_raw.json",
)
eval_output_path = os.path.join(
    "out", "eval", f"eval_llama-2-13b-chat-hf-300-v2-4-one_languid-stoat-38_v3.json"
)
trec_eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_llama-2-13b-chat-hf-300-v2-4-one_languid-stoat-38_v3_trec.txt",
)

eval_output_path = prepare_files(raw_eval_output_path, "super_duper_run")

scores = calculate_metrics(
    eval_output_path=eval_output_path,
    gold_labels_dir="data/gold_labels/",
    ft_model_name="llama-2-13b-chat-hf-300-v2-4-one",
    run_name="super_duper_run",
)
