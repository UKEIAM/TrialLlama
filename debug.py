import os
import json

from utils.eval_utils import (
    prepare_files,
    calculate_metrics,
    prepare_binary,
    evaluate_binary,
)

model_name = "llama-2-13b-chat-hf-3000-v7-5-v5-0.0001-1"
run_name = "wsbxtrqj-680_v7_1200_4_v5"

eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_{run_name}.json",
)

eval_df = prepare_files(eval_output_path, f"eval_{run_name}")

scores = calculate_metrics(
    eval_df=eval_df,
    gold_labels_dir="data/gold_labels/",
    ft_model_name=model_name,
    run_name=run_name,
)

print(scores)

eval_df = prepare_binary(eval_output_path, f"eval_{run_name}")

scores_binary = evaluate_binary(
    eval_df=eval_df,
    gold_labels_dir="data/gold_labels/",
    ft_model_name=model_name,
    run_name=run_name,
)

print(scores_binary)
print("FINISH")
#
# # Extract IDs to run base model on same CTs
# data_path = os.path.join("data", "ct_test_v9.json")
# data_path_out = os.path.join("data", "ct_test_v9_fixed_reasoning_set.json")
#
# # with open('out/eval/REASONING_lbn5d6tp-748_v9_50_4_v3.json', 'r') as file:
# #     data = json.load(file)
# #
# # with open(data_path, 'r') as test_file:
# #     test_data_full = json.load(test_file)
# #
# #
# # if isinstance(data, list):
# #     # Extract the "ID" entry of all items
# #     ids_to_filter = [item.get('ID') for item in data]
# #
# #     # Print or use the extracted IDs
# #     print(ids_to_filter)
# #     print(len(ids_to_filter))
# #
# #
# # filtered_data = [item for item in test_data_full if item.get("id") in ids_to_filter]
# #
# # with open(data_path_out, 'w') as filtered_file:
# #     json.dump(filtered_data, filtered_file, indent=2)
