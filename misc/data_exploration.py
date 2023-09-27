import os

import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(__file__))
type = "test"

path = base_dir
out_path = base_dir
path = os.path.join(base_dir, "data", f"ct_{type}_v3.json")

df = pd.read_json(path)


def count_words(text):
    # Split the text into words using whitespace as a separator and count them
    words = text.split()
    return len(words)


# Assuming you have a DataFrame named df with a "class" column
class_counts = df["output"].value_counts()

filtered_df = df[df["clinical_trial"].str.contains("Exclusion Criteria")]

class_counts_filtered = filtered_df["output"].value_counts()

# Step 4: Apply the function to your DataFrame column
cols_to_count = ["topic", "clinical_trial"]
filtered_df["word_count"] = filtered_df[cols_to_count].apply(
    lambda row: sum(row.map(count_words)), axis=1
)

max_words = filtered_df["word_count"].max()

mask = (
    filtered_df["word_count"] > 500
)  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
df_reduced = filtered_df[~mask]
class_counts_filtered_reduced = df_reduced["output"].value_counts()

print(class_counts_filtered_reduced)

# Create a bar chart of the class distribution
plt.figure(figsize=(8, 6))
ax = class_counts.plot(kind="bar", color="skyblue")
plt.title("Class Distribution of Clinical Trials Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)

for i, count in enumerate(class_counts_filtered_reduced):
    ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12, color="black")

# Save the plot as an image file (e.g., PNG)
plt.savefig(
    os.path.join(base_dir, "out", "eval", "img", f"class_distribution_{type}.png"),
    bbox_inches="tight",
)

# Check if the classes are balanced
is_balanced = all(class_counts.values == class_counts.values[0])
if is_balanced:
    print("The classes are balanced.")
else:
    print("The classes are not balanced.")
