import sys
import os
import re

import torch

from tqdm import tqdm
from pathlib import Path
from .memory_utils import MemoryTrace


sys.path.append(str(Path(__file__).resolve().parent.parent))


def test(
    model,
    test_set_json,
    test_config,
    test_dataloader,
    tokenizer
):
    """
    Run the model on a given test dataset. Returns a class 0, 1 or 2, which is saved to a
    .txt mapping the patient topic and the clinical trial ID.
    This is a preparation step to runt he trec_eval script.

    returns a .txt file.
    """

    pattern = r'^(\d+)_(\d+)_(\w+)\.(\w+)$'

    model.eval()
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(test_dataloader, colour="green", desc="evaluating Epoch")
        ):
            for key in batch.keys():
                batch[key] = batch[key].to("cuda:0")
            with torch.no_grad():
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, -1)

                match = re.match(pattern, test_set_json[step]['id'])
                if match:
                    internal_id = match.group(1)
                    topic_id = match.group(2)
                    ct_id = match.group(3)

                # tokens = tokenizer.batch_decode(
                #     preds.detach().cpu().numpy(), skip_special_tokens=True
                # )
