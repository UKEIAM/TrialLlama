import os

from utils.eval_utils import prepare_files, calculate_metrics

eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_xbf2eqdf-626_v7_4_0.0001_1.0.json",
)

eval_df = prepare_files(eval_output_path, "super_duper_run")

scores = calculate_metrics(
    eval_df=eval_df,
    gold_labels_dir="data/gold_labels/",
    ft_model_name="llama-2-13b-chat-hf-900-v7-4",
    run_name="debug_run",
)
