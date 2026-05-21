import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

input_path = os.path.join(BASE_DIR, "02_output", "cleaned_occupancy_log.csv")
output_path = os.path.join(BASE_DIR, "02_output", "smoothed_occupancy_log.csv")

df = pd.read_csv(input_path)

# Create temporary datetime columns for smoothing
df["timestamp_dt"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M:%S.%f"
)

df["date"] = df["timestamp_dt"].dt.date

# Sort so previous/next rows are correct
df = df.sort_values(["table", "date", "timestamp_dt"])

# Previous and next occupancy within each table and day
df["prev_occ"] = (
    df.groupby(["table", "date"])["occupancy"].shift(1)
)

df["next_occ"] = (
    df.groupby(["table", "date"])["occupancy"].shift(-1)
)

# Identify rows that will be changed
smooth_condition = (
    df["prev_occ"].notna()
    & df["next_occ"].notna()
    & (df["prev_occ"] == df["next_occ"])
    & (df["occupancy"] != df["prev_occ"])
)

# Output value of changes
n_changed = smooth_condition.sum()
prop_changed = smooth_condition.mean()

print("Number changed:", n_changed)
print("Proportion changed:", prop_changed)

# Update the original occupancy column
df.loc[smooth_condition, "occupancy"] = df.loc[smooth_condition, "prev_occ"]
df["occupancy"] = df["occupancy"].astype(int)

df = df.sort_values(["timestamp_dt", "table"])
df = df.drop(columns=["timestamp_dt", "date", "prev_occ", "next_occ"])
df = df[
    [
        "timestamp",
        "table",
        "occupancy",
        "dist_from_road",
        "in_shadow",
        "temp",
        "close_to_entry"
    ]
]

# Save
df.to_csv(output_path, index=False)