import sys
import os
import re
import random

import torch
import torch.nn.functional as F
import pandas as pd

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
    trec_out = pd.DataFrame(columns = ["TOPIC_NO", "Q0", "ID", "SCORE", "RUN_NAME"])
    df_out = pd.DataFrame(columns = ["TOPIC_NO", "Q0", "ID", "CLASS", "PROBA"]) # df_out is for internal evaluation purposes, where we can compare the output of the model with the qrels provided

    pattern = r"^(\d+)_(\d+)_(\w+)\.(\w+)$"

    model.eval()
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(test_dataloader, colour="green", desc="evaluating Epoch")
        ):
            for key in batch.keys():
                batch[key] = batch[key].to("cuda:0")
            with torch.no_grad():
                outputs = model(**batch)
                probs = F.softmax(outputs.logits, dim=1)
                class_0_probability = probs[:, 0]  # Probability for class "0"
                class_1_probability = probs[:, 1]  # Probability for class "1"
                class_2_probability = probs[:, 2]

                max_probs, predicted_labels = torch.max(probs, dim=1)
                max_probs = max_probs.cpu().detach().numpy()
                predicted_labels = predicted_labels.cpu().detach().numpy()

                match = re.match(pattern, test_set_json[step]["id"])
                if match:
                    internal_id = match.group(1)
                    topic_id = match.group(2)
                    ct_id = match.group(3)
                
                row_trec = [topic_id, 0, ct_id, random.random(), test_config.ft_model_name]
                row_out = [topic_id, 0, ct_id, max_probs, predicted_labels]
    
                # TODO: For debugging purposes, since currently model returns 'nan' values as output tensor
                trec_out.loc[step] = row_trec
                df_out.loc[step] = row_out

                # DEBUGING
                print(len(max_probs[0]))
                preds = torch.argmax(outputs.logits, -1)
                tokens = tokenizer.batch_decode(
                    preds.detach().cpu().numpy(), skip_special_tokens=True
                )
                print(tokens)
                print()

        trec_out["RANK"] = trec_out.groupby("TOPIC_NO")["SCORE"].rank(ascending=False, method="dense").astype(int)
        rank = trec_out.pop("RANK")
        trec_out.insert(3, "RANK", rank) # To be conform with the trec_eval format

        
        #add_ranking_column(trec_out)

        return trec_out, df_out


def add_ranking_column(df) -> pd.DataFrame:
    # TODO: Since trec_eval expects a ranking for a given topic-document combination, we need to create a rank topic-wise within our trec_out dataframe and add the ranks as new column entries
    df['RANK'] = df.groupby('TOPIC_ID')['SCORE'].rank(ascending=False, method='dense')

    return df