# TODO: Here functions are defined to evaluate the resulting file of the model_testing
import os
import mlflow

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
def calculate_metrics(
    eval_output_path: str,
    gold_labels_file: str,
    ft_model_name: str,
    run_name: str,
    logger,
):
    model_df = pd.read_csv(
        eval_output_path,
        header=None,
        delimiter="\t",
        names=["TOPIC", "Q0", "NCT_ID", "PROBA", "LABEL"],
    )
    model_df.drop(columns=["PROBA"], inplace=True)
    gold_df = pd.read_csv(
        gold_labels_file,
        header=None,
        delimiter=" ",
        names=["TOPIC", "Q0", "NCT_ID", "LABEL"],
    )

    # Merge the two dataframes on NCT_ID to filter for matching values
    merged_df = model_df.merge(gold_df, on="NCT_ID", suffixes=("_pred", "_gold"))

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
        y_true = label_binarize(merged_df["LABEL_gold"], classes=[0, 1, 2])
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

    # Plot the confusion matrix TODO: Not working  yet
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["irrelevant", "uneligible", "eligible"],
        yticklabels=["irrelevant", "uneligible", "eligible"],
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
