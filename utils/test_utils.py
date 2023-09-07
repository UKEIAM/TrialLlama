import sys
import os

import torch
import tqdm

from pathlib import Path
from .memory_utils import MemoryTrace





sys.path.append(str(Path(__file__).resolve().parent.parent))

def test(model, test_set_json, train_config, test_dataloader, local_rank, tokenizer, input_json):
    """
    Run the model on a given test dataset. Returns a class 0, 1 or 2, which is saved to a
    .txt mapping the patient topic and the clinical trial ID.
    This is a preparation step to runt he trec_eval script.

    returns a .txt file.
    """

    # TODO: Get json 'input' key and separate topic and CT ID with ^([^_]+)_([^_]+)_([^_]+)\.xml$

    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(test_dataloader, colour="green", desc="evaluating Epoch")
        ):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to("cuda:0")
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)

                predictions = outputs.logits.argmax(dim=-1)

                