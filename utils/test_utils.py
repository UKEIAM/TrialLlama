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


sys.path.append(str(Path(__file__).resolve().parent.parent))


def test(
    model, test_data_json, test_config, test_dataloader, tokenizer, max_tokens
) -> pd.DataFrame:
    """
    Run the model on a given test dataset. Returns a class 0, 1 or 2, which is saved to a
    .txt mapping the patient topic and the clinical trial ID.
    This is a preparation step to runt he trec_eval script.

    returns a .txt file.
    """
    trec_out = pd.DataFrame(columns=["TOPIC_NO", "Q0", "ID", "SCORE", "RUN_NAME"])
    df_out = pd.DataFrame(
        columns=["TOPIC_NO", "Q0", "ID", "CLASS", "PROBA"]
    )  # df_out is for internal evaluation purposes, where we can compare the output of the model with the qrels provided

    id_pattern = r"^(\d+)_(\d+)_(\w+)$"
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
                        top_p=test_config.top_p,
                        temperature=test_config.temperature,
                        min_length=test_config.min_length,
                        use_cache=test_config.use_cache,
                        top_k=test_config.top_k,
                        repetition_penalty=test_config.repetition_penalty,
                        length_penalty=test_config.length_penalty,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                except RuntimeError as e:
                    print(e)
                    print(test_data_json[step]["id"])
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
                response = []
                for token in generated_tokens[0]:
                    response.append(tokenizer.decode(token))
                response = "".join(response)
                response = response.replace("</s>", "")
                if test_config.debug:
                    print(f"### Response: {response}")

                match = re.match(id_pattern, test_data_json[step]["id"])
                internal_id = match.group(1)
                topic_id = match.group(2)
                ct_id = match.group(3)

                probas = []
                if "uneligible" in response.lower():
                    for item in transition_scores[0]:
                        probas.append(np.exp(item.cpu().numpy()))
                    proba = sum(probas) / len(probas)
                    predicted_label = 1
                elif "eligible" in response.lower():
                    for item in transition_scores[0]:
                        probas.append(np.exp(item.cpu().numpy()))
                    proba = sum(probas) / len(probas)
                    predicted_label = 2
                elif "irrelevant" in response.lower():
                    for item in transition_scores[0]:
                        probas.append(np.exp(item.cpu().numpy()))
                    proba = sum(probas) / len(probas)
                    predicted_label = 0
                else:
                    empty_response_counter += 1
                    if test_config.debug:
                        # TODO: Currently, model response is often gibberish or nothing at all. Don't know yet how to handle such values
                        # TODO: Has to be considered in evaluation, since less examples are evaluated between the models...
                        print(f"Internal ID: {match.group(1)}")

                        print(
                            "Response gibberish or empty. Continuing to next example."
                        )
                    continue

                row_trec = [topic_id, 0, ct_id, proba, test_config.ft_model]
                row_out = [topic_id, 0, ct_id, proba, predicted_label]

                # TODO: For debugging purposes, since currently model returns 'nan' values as output tensor
                trec_out.loc[step] = row_trec
                df_out.loc[step] = row_out

        # trec_eval script requires a document ranking. For that reason we simply use score calculated by the averaged token probablities to create a doucment ranking
        trec_out["RANK"] = (
            trec_out.groupby("TOPIC_NO")["SCORE"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )
        rank = trec_out.pop("RANK")
        trec_out.insert(3, "RANK", rank)  # To be conform with the trec_eval format

        # add_ranking_column(trec_out)

        return trec_out, df_out, empty_response_counter


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
