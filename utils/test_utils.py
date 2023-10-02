import sys
import os
import re
import random

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from .memory_utils import MemoryTrace
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))


def test(
    model,
    test_data_json,
    test_config,
    test_dataloader,
    tokenizer,
    max_tokens,
    logger: Optional[object] = None,
) -> tuple:
    """
    Run the model on a given test dataset. Returns a class 0, 1 or 2, which is saved to a
    .txt mapping the patient topic and the clinical trial ID.
    This is a preparation step to runt he trec_eval script.

    returns a .txt file.
    """
    raw_out = pd.DataFrame(columns=["ID", "TOPIC_YEAR", "RESPONSE", "PROBA"])
    empty_response_counter = 0

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(test_dataloader, colour="green", desc="Testing iteration:")
        ):
            for key in batch.keys():
                batch[key] = batch[key].view(1, max_tokens)
                batch[key] = batch[key].to("cuda:0")
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **batch,
                        max_new_tokens=test_config.max_new_tokens,
                        do_sample=test_config.do_sample,
                        temperature=test_config.temperature,
                        min_length=test_config.min_length,
                        use_cache=test_config.use_cache,
                        top_k=test_config.top_k,
                        top_p=test_config.top_p,
                        repetition_penalty=test_config.repetition_penalty,
                        length_penalty=test_config.length_penalty,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                except RuntimeError as e:
                    logger.error(
                        f"Model eval Output Error: {e} | Sample Identifier {test_data_json[step]['id']}"
                    )
                    continue

                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                input_length = (
                    1
                    if model.config.is_encoder_decoder
                    else batch["input_ids"].shape[1]
                )
                generated_tokens = outputs.sequences[:, input_length:]

                resp_sentences = []
                current_word = ""
                special_symbol_map = {
                    "<0x0A>": "\n",  # Replace <0x0A> with a newline character
                    # Add more mappings as needed
                }
                # for token in generated_tokens[0]:
                #     token = tokenizer.decode(token)
                #     # Check if the token starts with a space, indicating a subword token
                #     if token in special_symbol_map:
                #         token = special_symbol_map[token]
                #     if token.startswith(" "):
                #         current_word += token
                #     else:
                #         if current_word:
                #             # Add the current word to the list of sentences
                #             resp_sentences.append(current_word)
                #         current_word = token
                # Add the last word if it exists
                # if current_word:
                #     resp_sentences.append(current_word)
                response = tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=True
                )
                # response = " ".join(resp_sentences)
                # response = response.replace("</s>", "")

                if test_config.debug:
                    print(f"{response}")

                topic_year = test_data_json[step]["topic_year"]
                probas = []
                for item in transition_scores[0]:
                    probas.append(np.exp(item.cpu().numpy()))
                proba = sum(probas) / len(probas)

                if "" in response.lower():
                    empty_response_counter += 1

                row_raw = [
                    test_data_json[step]["id"],
                    topic_year,
                    response,
                    proba,
                ]
                raw_out.loc[step] = row_raw

        return raw_out, empty_response_counter


def get_max_length(model):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """

    # Pull model configuration
    conf = model.config
    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length
