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

from typing import Optional

# Load the model-output and gold-labels files
def prepare_files(
    eval_output_path, run_name, logger: Optional[object] = None
) -> pd.DataFrame:
    # TODO Prepare the trec_eval output file and save it as well as the output file required to run metrics
    # Original columns: ["ID", "TOPIC_YEAR", "RESPONSE", "PROBA"]
    raw_df = pd.read_json(eval_output_path, orient="records")
    id_pattern = r"^(\d+)_(\d+\-\d+)_(\w+)$"
    pattern = r"[A-C][:. ]?\s?\(?\w+\)?|[A-C]\s"  # Pattern includes answers like A: eligible, B excluded and C (not relevant)

    eval_df = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "LABEL", "TOPIC_YEAR"])

    trec_eval = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "SCORE", "RUN_NAME"])

    for item in raw_df.iterrows():
        match = re.match(id_pattern, item[1]["ID"])
        match_label = re.findall(pattern, item[1]["RESPONSE"])
        if len(match_label) == 0 or len(match_label) > 1:
            continue
        resp = match_label[0].lower()

        if "excluded" in resp or "B" in resp:
            pred_class = 1
        elif "eligible" in resp or "A" in resp:
            pred_class = 2
        elif "irrelevant" in resp or "C" in resp:
            pred_class = 0
        else:
            continue

        # Create a dictionary representing the new row
        try:
            new_row_eval = {
                "TOPIC_NO": int(match.group(2).split("-")[0]),
                "Q0": 0,
                "NCT_ID": match.group(3),
                "LABEL": int(pred_class),
                "TOPIC_YEAR": int(item[1]["TOPIC_YEAR"]),
            }
            new_row_trec = {
                "TOPIC_NO": match.group(2).split("-")[0],
                "Q0": 0,
                "NCT_ID": match.group(3),
                "SCORE": item[1]["PROBA"],
                "RUN_NAME": run_name,
                "TOPIC_YEAR": item[1]["TOPIC_YEAR"],
            }
        except UnboundLocalError as e:
            if logger != None:
                logger.error(f"Response: {resp} -> {e}")
            continue

        # Convert the new row into a DataFrame
        new_row_eval_df = pd.DataFrame([new_row_eval])
        new_row_trec_df = pd.DataFrame([new_row_trec])

        # TODO: Figure out if trec_eval script can differentiateFor trec_eval split df_raw by year and create two seperate files for the years!
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

    """
        As trec_eval operates by year, we need to split our mixed test data by the topic year and save the resulting
        files separately.
    """
    unique_years = trec_eval["TOPIC_YEAR"].unique()
    sub_dataframes = {}
    for year in unique_years:
        sub_df = trec_eval[
            trec_eval["TOPIC_YEAR"] == year
        ].copy()  # Create a copy to avoid modifying the original DataFrame
        sub_df.drop(columns=["TOPIC_YEAR"], inplace=True)  # Drop the 'YEAR' column
        sub_dataframes[year] = sub_df

    for year, sub_df in sub_dataframes.items():
        trec_eval_path = eval_output_path.replace("eval/", "eval/trec_eval/")
        os.makedirs(os.path.dirname(trec_eval_path), exist_ok=True)
        trec_eval_path = trec_eval_path.replace(".json", f"_trec_{int(year)}.json")
        sub_df.to_csv(f"{trec_eval_path}", sep="\t", header=False, index=False)

    """
        The dataframe required for metrics calculation is only saved during runtime and not permanently.
    """

    return eval_df


def calculate_metrics(
    eval_df: pd.DataFrame,
    gold_labels_dir: str,
    ft_model_name: str,
    run_name: str,
    logger: Optional[object] = None,
):
    year_pattern = r"(\d{4})\.txt$"
    gold_dfs = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "LABEL", "TOPIC_YEAR"])
    for filename in os.listdir(gold_labels_dir):
        gold_df = pd.read_csv(
            os.path.join(gold_labels_dir, filename),
            header=None,
            delimiter=" ",
            names=["TOPIC_NO", "Q0", "NCT_ID", "LABEL"],
        )
        match = re.search(year_pattern, filename)
        year = int(match.group(1))
        gold_df["TOPIC_YEAR"] = year
        gold_dfs = pd.concat([gold_dfs, gold_df], ignore_index=True)

    # Merge the two dataframes on NCT_ID to filter for matching values
    """
         Very funny bug: all dtypes of eval_df are object. Such are gold_dfs. Nevertheless, merging on TOPIC_NO,
         merged_df becomes empty. If removing TOPIC_NO, merge works fine.
         If df is saved to json and the imported with pd.read_json(), dtypes of most columns is int64. Merge works with
         TOPIC_NO included. So transforming the dtype object to int64 in the eval_df, fixes the problem as well.
    """
    eval_df["LABEL"] = eval_df["LABEL"].astype(int)
    gold_dfs["LABEL"] = gold_dfs["LABEL"].astype(int)
    merged_df = eval_df.merge(
        gold_dfs, on=["TOPIC_NO", "NCT_ID", "TOPIC_YEAR"], suffixes=("_pred", "_gold")
    )
    try:
        assert len(merged_df) == len(eval_df)
    except AssertionError as e:
        if logger != None:
            logger.error(e)
    # Calculate Accuracy, F1 score, and AUC
    accuracy = accuracy_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"])
    precision = precision_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    recall = recall_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    f1 = f1_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro")
    p_at_5 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 5)
    p_at_10 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 10)
    p_at_50 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 50)

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

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        # cmap="Blues",
        xticklabels=["irrelevant", "excluded", "eligible"],
        yticklabels=["irrelevant", "excluded", "eligible"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    out_img_path = os.path.join("out", "eval", "img")
    os.makedirs(out_img_path, exist_ok=True)
    plt.savefig(os.path.join(out_img_path, f"cm_{ft_model_name}_{run_name}.png"))

    return {
        "evaluatable_items": len(merged_df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "p_at_5": p_at_5,
        "p_at_10": p_at_10,
        "p_at_50": p_at_50,
    }


def precision_at_k(true_items, predicted_items, k):
    # Ensure both lists have a length of at least k
    true_items = true_items[:k]
    predicted_items = predicted_items[:k]

    # Calculate the intersection of true_items and predicted_items
    intersection = set(true_items) & set(predicted_items)

    # Calculate precision at K
    precision = len(intersection) / k if k > 0 else 0.0

    return precision


def prepare_binary(
    eval_output_path, run_name, logger: Optional[object] = None
) -> pd.DataFrame:
    # TODO Prepare the trec_eval output file and save it as well as the output file required to run metrics
    # Original columns: ["ID", "TOPIC_YEAR", "RESPONSE", "PROBA"]
    raw_df = pd.read_json(eval_output_path, orient="records")
    id_pattern = r"^(\d+)_(\d+\-\d+)_(\w+)$"
    pattern = r"[A-C][:. ]?\s?\(?\w+\)?|[A-C]\s"  # Pattern includes answers like A: eligible, B excluded and C (not relevant)

    eval_df = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "LABEL", "TOPIC_YEAR"])

    for item in raw_df.iterrows():
        match = re.match(id_pattern, item[1]["ID"])
        match_label = re.findall(pattern, item[1]["RESPONSE"])
        if len(match_label) == 0 or len(match_label) > 1:
            continue
        resp = match_label[0].lower()

        if "excluded" in resp or "B" in resp:
            pred_class = 1
        elif "eligible" in resp or "A" in resp:
            pred_class = 2
        elif "irrelevant" in resp or "C" in resp:
            pred_class = 1
        else:
            continue

        # Create a dictionary representing the new row
        try:
            new_row_eval = {
                "TOPIC_NO": int(match.group(2).split("-")[0]),
                "Q0": 0,
                "NCT_ID": match.group(3),
                "LABEL": int(pred_class),
                "TOPIC_YEAR": int(item[1]["TOPIC_YEAR"]),
            }

        except UnboundLocalError as e:
            if logger != None:
                logger.error(f"Response: {resp} -> {e}")
            continue

        # Convert the new row into a DataFrame
        new_row_eval_df = pd.DataFrame([new_row_eval])

        # TODO: Figure out if trec_eval script can differentiateFor trec_eval split df_raw by year and create two seperate files for the years!
        # Concatenate the new row DataFrame with the original eval_df
        eval_df = pd.concat([eval_df, new_row_eval_df], ignore_index=True)

    return eval_df


def evaluate_binary(
    eval_df: pd.DataFrame,
    gold_labels_dir: str,
    ft_model_name: str,
    run_name: str,
    logger: Optional[object] = None,
):
    year_pattern = r"(\d{4})\.txt$"
    gold_dfs = pd.DataFrame(columns=["TOPIC_NO", "Q0", "NCT_ID", "LABEL", "TOPIC_YEAR"])
    for filename in os.listdir(gold_labels_dir):
        gold_df = pd.read_csv(
            os.path.join(gold_labels_dir, filename),
            header=None,
            delimiter=" ",
            names=["TOPIC_NO", "Q0", "NCT_ID", "LABEL"],
        )
        match = re.search(year_pattern, filename)
        year = int(match.group(1))
        gold_df["TOPIC_YEAR"] = year
        gold_dfs = pd.concat([gold_dfs, gold_df], ignore_index=True)

    # Merge the two dataframes on NCT_ID to filter for matching values
    """
         Very funny bug: all dtypes of eval_df are object. Such are gold_dfs. Nevertheless, merging on TOPIC_NO,
         merged_df becomes empty. If removing TOPIC_NO, merge works fine.
         If df is saved to json and the imported with pd.read_json(), dtypes of most columns is int64. Merge works with
         TOPIC_NO included. So transforming the dtype object to int64 in the eval_df, fixes the problem as well.
    """
    condition = gold_dfs["LABEL"] == 0
    gold_dfs.loc[condition, "LABEL"] = 1
    eval_df["LABEL"] = eval_df["LABEL"].astype(int)
    gold_dfs["LABEL"] = gold_dfs["LABEL"].astype(int)
    merged_df = eval_df.merge(
        gold_dfs, on=["TOPIC_NO", "NCT_ID", "TOPIC_YEAR"], suffixes=("_pred", "_gold")
    )
    try:
        assert len(merged_df) == len(eval_df)
    except AssertionError as e:
        if logger != None:
            logger.error(e)
    # Calculate Accuracy, F1 score, and AUC
    accuracy = accuracy_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"])
    precision = precision_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    recall = recall_score(
        merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro"
    )
    f1 = f1_score(merged_df["LABEL_gold"], merged_df["LABEL_pred"], average="macro")
    p_at_5 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 5)
    p_at_10 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 10)
    p_at_50 = precision_at_k(merged_df["LABEL_gold"], merged_df["LABEL_pred"], 50)

    try:
        y_true = merged_df["LABEL_gold"]
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
        # cmap="Blues",
        xticklabels=["excluded", "eligible"],
        yticklabels=["excluded", "eligible"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    out_img_path = os.path.join("out", "eval", "img")
    os.makedirs(out_img_path, exist_ok=True)
    plt.savefig(os.path.join(out_img_path, f"cm_{ft_model_name}_{run_name}.png"))

    return {
        "binary_accuracy": accuracy,
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_auc": auc,
        "binary_p_at_5": p_at_5,
        "binary_p_at_10": p_at_10,
        "binary_p_at_50": p_at_50,
    }
