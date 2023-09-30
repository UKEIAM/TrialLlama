import os

from utils.eval_utils import prepare_files

raw_eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_llama-2-13b-chat-hf-300-v2-4-None_polite-shrimp-650_v3_raw.json",
)
eval_output_path = os.path.join(
    "out", "eval", f"eval_llama-2-13b-chat-hf-300-v2-4-None_polite-shrimp-650_v3.json"
)
trec_eval_output_path = os.path.join(
    "out",
    "eval",
    f"eval_llama-2-13b-chat-hf-300-v2-4-None_polite-shrimp-650_v3_trec.txt",
)

prepare_files(
    raw_eval_output_path, eval_output_path, trec_eval_output_path, "super_duper_run"
)
