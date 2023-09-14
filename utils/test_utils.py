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


def test(model, test_set_json, test_config, test_dataloader, tokenizer) -> pd.DataFrame:
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
    response_pattern = r"### Response:(.*)"

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(test_dataloader, colour="green", desc="Testing iteration:")
        ):
            for key in batch.keys():
                batch[key] = batch[key].view(1, test_config.max_tokens)
                batch[key] = batch[key].to("cuda:0")
            with torch.no_grad():
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
                # output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                # TODO: Only consider part after "###Response: "

                transition_scores = model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                input_length = (
                    1
                    if model.config.is_encoder_decoder
                    else batch["input_ids"].shape[1]
                )
                generated_tokens = outputs.sequences[:, input_length:]
                response = tokenizer.decode(
                    generated_tokens[0][idx]
                )  # TODO: Right now, assuming Model returns only the class.
                for idx, tokens in enumerate(generated_tokens):
                    response.append(tokenizer.decode(generated_tokens[0][idx]))

                for idx, item in enumerate(response):
                    if "eligible" in item.lower():
                        proba = np.exp(transition_scores[0][idx].cpu().numpy())
                        predicted_label = 2
                    elif "noneligible" in item.lower():
                        proba = np.exp(transition_scores[0][idx].cpu().numpy())
                        predicted_label = 1
                    elif "irrelevant" in item.lower():
                        proba = np.exp(transition_scores[0][idx].cpu().numpy())
                        predicted_label = 0
                    else:
                        proba = None
                        predicted_label = -1

                match = re.match(id_pattern, test_set_json[step]["id"])
                internal_id = match.group(1)
                topic_id = match.group(2)
                ct_id = match.group(3)

                print(f"### Response: {response}")
                # TODO: Figure out how to get probabilities for the output of LLM
                if predicted_label in [0, 1, 2]:
                    row_trec = [topic_id, 0, ct_id, proba, test_config.ft_model_name]
                    row_out = [topic_id, 0, ct_id, proba, predicted_label]

                # TODO: For debugging purposes, since currently model returns 'nan' values as output tensor
                trec_out.loc[step] = row_trec
                df_out.loc[step] = row_out

        trec_out["RANK"] = (
            trec_out.groupby("TOPIC_NO")["SCORE"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )
        rank = trec_out.pop("RANK")
        trec_out.insert(3, "RANK", rank)  # To be conform with the trec_eval format

        # add_ranking_column(trec_out)

        return trec_out, df_out
