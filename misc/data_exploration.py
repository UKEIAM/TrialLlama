import os
import re

import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(__file__))
type = "test"
one_shot = True
max_allowed_words = 1500
binary_balancing = True
path = base_dir
out_path = base_dir
path = os.path.join(base_dir, "data", f"ct_{type}_v7.json")

df = pd.read_json(path)

print(f"DATASET LENGTH: {len(df)}")


def count_words(text):
    # Split the text into words using whitespace as a separator and count them
    words = text.split()
    return len(words)


# Assuming you have a DataFrame named df with a "class" column
class_counts = df["output"].value_counts()
print(class_counts)


# Step 4: Apply the function to your DataFrame column
cols_to_count = ["instruction", "topic", "clinical_trial", "response"]
df["word_count"] = df[cols_to_count].apply(
    lambda row: sum(row.map(count_words)), axis=1
)

max_words = df["word_count"].max()

mask = (
    df["word_count"] > max_allowed_words
)  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
df_reduced = df[~mask]
class_counts_filtered_reduced = df_reduced["output"].value_counts()

print(class_counts_filtered_reduced)

# CURATED DATASET: Reduce amount of data in returning only x examples per patient topic

df_reduced["topic_id"] = df_reduced["id"].str.split("_").str[1]
balanced_df = pd.DataFrame(columns=df_reduced.columns)

for unique_id in df_reduced["topic_id"].unique():
    id_subset = df_reduced[df_reduced["topic_id"] == unique_id]
    desired_label_count = (
        id_subset.groupby("topic_id")["output"].value_counts().sort_values().iloc[0]
    )
    # Get all rows with the current unique ID
    id_subset = df_reduced[df_reduced["topic_id"] == unique_id]

    # Separate the rows by label
    label_groups = [
        id_subset[id_subset["output"] == label]
        for label in id_subset["output"].unique()
    ]

    if binary_balancing:
        try:
            aspired_total_sampels = desired_label_count * 3
            len_eligible = len(id_subset[id_subset["output"] == "A: eligible"])
            eligible_samples = (
                len_eligible
                if len_eligible < int(aspired_total_sampels / 2)
                else int(aspired_total_sampels / 2)
            )
            eligible = id_subset[id_subset["output"] == "A: eligible"].sample(
                eligible_samples
            )

            len_excluded = len(id_subset[id_subset["output"] == "B: excluded"])
            excluded_samples = (
                len_excluded
                if len_eligible < int(aspired_total_sampels / 4)
                else int(aspired_total_sampels / 4)
            )
            excluded = id_subset[id_subset["output"] == "B: excluded"].sample(
                excluded_samples
            )

            len_irrelevant = len(id_subset[id_subset["output"] == "C: irrelevant"])
            irrelevant_sample = (
                len_irrelevant
                if len_irrelevant < int(aspired_total_sampels / 4)
                else int(aspired_total_sampels / 4)
            )
            irrelevant = id_subset[id_subset["output"] == "C: irrelevant"].sample(
                irrelevant_sample
            )

            balanced_df = pd.concat(
                [balanced_df, eligible, excluded, irrelevant], ignore_index=True
            )
        except ValueError as e:
            print("VALUE COUNTS 0: One of the grouped labels is 0. Continuing.")
            continue
    else:
        balanced_label_groups = [
            group.sample(n=len(group), random_state=42)
            if len(group) < desired_label_count
            else group.sample(n=desired_label_count, random_state=42)
            for group in label_groups
        ]

        # Concatenate the balanced label groups and add them to the balanced_df
        for df_item in balanced_label_groups:
            balanced_df = pd.concat([balanced_df, df_item], ignore_index=True)

value_counts_grouped = balanced_df["output"].value_counts()

if one_shot:
    example_path = os.path.join(base_dir, "data", f"inference_example_v6.json")
    max_words_2 = balanced_df["word_count"].max()
    df_example = pd.read_json(example_path)
    input_value = df_example["input"]
    balanced_df["instruction"] = (
        balanced_df["instruction"].astype("str") + input_value[0]
    )

for col in cols_to_count:
    balanced_df[col] = balanced_df[col].astype("str")

balanced_df["word_count"] = balanced_df[cols_to_count].apply(
    lambda row: sum(row.map(count_words)), axis=1
)

max_words_3 = balanced_df["word_count"].max()
mask = (
    balanced_df["word_count"] > max_allowed_words
)  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
df_reduced_final = balanced_df[~mask]
class_counts_filtered_reduced_final = df_reduced_final["output"].value_counts()

print(class_counts_filtered_reduced_final)

# Create a bar chart of the class distribution
plt.figure(figsize=(8, 6))
ax = class_counts_filtered_reduced_final.plot(kind="bar", color="skyblue")
plt.title("Class Distribution of Clinical Trials Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)

for i, count in enumerate(class_counts_filtered_reduced_final):
    ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12, color="black")

# Save the plot as an image file (e.g., PNG)
plt.savefig(
    os.path.join(
        base_dir,
        "out",
        "eval",
        "img",
        f"class_distribution_{type}_{max_allowed_words}.png",
    ),
    bbox_inches="tight",
)
