# TODO: Here functions are defined to evaluate the resulting file of the model_testing
import os
import mlflow

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Load the model-output and gold-labels files
def calculate_metrics(eval_output_path, gold_labels_file, ft_model_name):
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
    f1 = f1_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="weighted")
    try:
        auc = roc_auc_score(
            merged_df["LABEL_gold"],
            merged_df["LABEL_pred"],
            average="weighted",
            multi_class="ovr",
        )
    except Exception as e:
        # TODO: Delete, for debug reasons only!
        auc = 0

    # Create a confusion matrix
    conf_matrix = confusion_matrix(merged_df["LABEL_gold"], merged_df["LABEL_pred"])

    # Plot the confusion matrix TODO: Not working  yet
    print(conf_matrix)
    print(type(conf_matrix))
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
    plt.savefig(os.path.join(out_img_path, f"cm_{ft_model_name}.png"))

    return {"accuracy": accuracy, "f1": f1, "auc": auc}
