import os

from utils.eval_utils import (
    prepare_files,
    calculate_metrics,
    prepare_binary,
    evaluate_binary,
)

eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_uyujjvjy-931_v7_1000_4_v3.json",
)

eval_df = prepare_files(eval_output_path, "super_duper_run")

scores = calculate_metrics(
    eval_df=eval_df,
    gold_labels_dir="data/gold_labels/",
    ft_model_name="llama-2-13b-chat-hf-3000-v7-4-v3",
    run_name="debug_run",
)

print(scores)

eval_df = prepare_binary(eval_output_path, "super_duper_run")

scores_binary = evaluate_binary(
    eval_df=eval_df,
    gold_labels_dir="data/gold_labels/",
    ft_model_name="llama-2-13b-chat-hf-3000-v7-4-v3",
    run_name="debug_run",
)

print(scores_binary)
print("FINISH")
