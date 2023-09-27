import os

import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(__file__))

type = "train"

path = base_dir
out_path = base_dir
if type == "train":
    path = os.path.join(base_dir, "data", "ct_full_v3.json")
elif type == "test":
    path = os.path.join(base_dir, "data", "ct_testing_full_v3.json")

df = pd.read_json(path)

# Assuming you have a DataFrame named df with a "class" column
class_counts = df["output"].value_counts()

filtered_df = df[df["input"].str.contains("Exclusion Criteria")]

class_counts_filtered = filtered_df["output"].value_counts()

# Step 3: Define a function to count words
def count_words(text):
    # Split the text into words using whitespace as a separator and count them
    words = text.split()
    return len(words)


# Step 4: Apply the function to your DataFrame column
filtered_df["word_count"] = filtered_df["input"].apply(count_words)

max_words = filtered_df["word_count"].max()


mask = (
    filtered_df["word_count"] > 600
)  # Checking the data on random samples showed that most inputs wich have more than 500 words are gibberish since the trial did not keep a proper format that is processable by the system.
df_reduced = filtered_df[~mask]
class_counts_filtered_reduced = df_reduced["output"].value_counts()


# Create a bar chart of the class distribution
plt.figure(figsize=(8, 6))
ax = class_counts.plot(kind="bar", color="skyblue")
plt.title("Class Distribution of Clinical Trials Dataset")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)

for i, count in enumerate(class_counts):
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
