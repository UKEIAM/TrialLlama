# TODO: Here functions are defined to evaluate the resulting file of the model_testing
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

# Load the model-output and gold-labels files
def prepare_files(raw_eval_path, eval_path, trec_eval_path, run_name):
    # TODO Prepare the trec_eval output file and save it as well as the output file required to run metrics
    # Original columns: ["ID", "TOPIC_YEAR", "RESPONSE", "PROBA"]
    raw_df = pd.read_json(raw_eval_path, orient="records")
    id_pattern = r"^(\d+)_(\d+)_(\w+)$"

    eval_df = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "LABEL"])

    trec_eval = pd.DataFrame(columns=["TOPIC_NO", "Q0", "ID", "SCORE", "RUN_NAME"])

    for item in raw_df.iterrows():
        match = re.match(id_pattern, item[1]["ID"])
        resp = item[1]["RESPONSE"].lower()
        resp = "".join(resp.split())

        if "noteligible" in resp:
            pred_class = 1
        elif "eligible" in resp:
            pred_class = 2
        elif "norelevantinformation" in resp:
            pred_class = 0

        # Create a dictionary representing the new row
        new_row_eval = {
            "TOPIC_NO": match.group(1),
            "Q0": 0,
            "NCT_ID": match.group(3),
            "LABEL": pred_class,
        }
        new_row_trec = {
            "TOPIC_NO": match.group(1),
            "Q0": 0,
            "ID": match.group(3),
            "SCORE": item[1]["PROBA"],
            "RUN_NAME": run_name,
        }

        # Convert the new row into a DataFrame
        new_row_eval_df = pd.DataFrame([new_row_eval])
        new_row_trec_df = pd.DataFrame([new_row_trec])

        # Concatenate the new row DataFrame with the original eval_df
        eval_df = pd.concat([eval_df, new_row_eval_df], ignore_index=True)
        trec_eval = pd.concat([trec_eval, new_row_trec_df], ignore_index=True)

    trec_eval["RANK"] = (
        trec_eval.groupby("TOPIC_NO")["SCORE"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    rank = trec_eval.pop("RANK")
    trec_eval.insert(3, "RANK", rank)

    eval_df.to_json(eval_path, orient="records")
    trec_eval.to_csv(trec_eval_path, sep="\t", header=False, index=False)

    # return eval_df


def calculate_metrics(
    eval_output_path: str,
    gold_labels_file: str,
    ft_model_name: str,
    run_name: str,
    logger,
):
    df = pd.read_json(eval_output_path)

    gold_df = pd.read_csv(
        gold_labels_file,
        header=None,
        delimiter=" ",
        names=["TOPIC_NO", "Q0", "NCT_ID", "LABEL"],
    )

    # Merge the two dataframes on NCT_ID to filter for matching values
    merged_df = df.merge(
        gold_df, on=["TOPIC_NO", "NCT_ID"], suffixes=("_pred", "_gold")
    )

    # Calculate Accuracy, F1 score, and AUC
    accuracy = accuracy_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"])
    precision = precision_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    recall = recall_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    f1 = f1_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro")
    try:
        y_true = label_binarize(merged_df["LABEL_gold"], labeles=[0, 1, 2])
        y_pred = merged_df[["LABEL_pred"]]
        auc = roc_auc_score(
            y_true,
            y_pred,
            average="macro",
            multi_class="ovr",
        )
    except Exception as e:
        # TODO: Delete, for debug reasons only!
        logger.error(f"AUC error occured: {e}")
        auc = 0

    # Create a confusion matrix
    conf_matrix = confusion_matrix(merged_df["LABEL_gold"], merged_df["LABEL_pred"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["no relevant information", "not eligible", "eligible"],
        yticklabels=["no relevant information", "not eligible", "eligible"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    out_img_path = os.path.join("out", "eval", "img")
    os.makedirs(out_img_path, exist_ok=True)
    plt.savefig(os.path.join(out_img_path, f"cm_{ft_model_name}_{run_name}.png"))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }
