import json


def count_tokens(dataset_path, tokenizer):
    PROMPT_DICT = {
        "prompt_input": (
            "{instruction}\n\n{topic}\n\n{clinical_trial}\n\n{response}\n\n"
        )
    }
    ann = json.load(open(dataset_path))
    total_tokens = 0
    for item in ann:
        prompt = PROMPT_DICT["prompt_input"].format_map(item)
        tokens = tokenizer(prompt)
        total_tokens += len(tokens[0])

    return total_tokens
