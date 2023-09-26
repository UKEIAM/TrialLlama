import os

import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(__file__))

type = "train"

path = base_dir
out_path = base_dir
if type == "train":
    path = os.path.join(base_dir, "data", "ct_full_v2.json")
    out_path = os.path.join(base_dir, "data", "ct_v2.json")
elif type == "test":
    path = os.path.join(base_dir, "data", "ct_testing_full_v2.json")
    out_path = os.path.join(base_dir, "data", "ct_testing.json")

df = pd.read_json(path)

# Assuming you have a DataFrame named df with a 'class' column
class_counts = df["output"].value_counts()

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
