import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(BASE_DIR, "02_output", "occupancy_log.csv")
output_path = os.path.join(BASE_DIR, "02_output", "cleaned_occupancy_log.csv")

df = pd.read_csv(input_path)

##############################
# Time fix
##############################
df.loc[0:531, "timestamp"] = df.loc[0:531, "timestamp"].str.replace(
    r"^13-05-2026",
    "12-05-2026",
    regex=True
)

start_times = pd.DataFrame({
    "recording_day": pd.to_datetime([
        "2026-05-12",
        "2026-05-13",
        "2026-05-15"
    ]).date,
    "actual_start": pd.to_datetime([
        "2026-05-12 12:16:39",
        "2026-05-13 12:03:45",
        "2026-05-15 12:13:58"
    ])
})

df["timestamp_dt"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M:%S.%f"
)

df["recording_day"] = df["timestamp_dt"].dt.date

df["time_index"] = (
    df.groupby("recording_day")["timestamp_dt"]
      .transform(lambda x: x.map({
          time: i for i, time in enumerate(sorted(x.unique()))
      }))
)

df = df.merge(start_times, on="recording_day", how="left")

df["timestamp_dt"] = df["actual_start"] + pd.to_timedelta(
    df["time_index"] * 30,
    unit="s"
)

df["timestamp"] = df["timestamp_dt"].dt.strftime("%d-%m-%Y %H:%M:%S.000")

df = df.drop(
    columns=["recording_day", "actual_start", "time_index", "timestamp_dt"]
)

##############################
# Close to entry
##############################
df["close_to_entry"] = np.where(
    df["table"].isin(["table_3", "table_4"]),
    True,
    False
)

df["close_to_entry"] = df["close_to_entry"].astype("category")

##############################
# Distance from road
##############################
distance_map = {
    "table_1": 5.5,
    "table_2": 8.6,
    "table_3": 10.5,
    "table_4": 10.5
}

df["dist_from_road"] = df["table"].map(distance_map).fillna(df["dist_from_road"])

##############################
# In shadow by date and table
##############################

df["date"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M:%S.%f"
).dt.date

conditions = [
    (df["date"] == pd.to_datetime("2026-05-12").date()) & (df["table"] == "table_1"),
    (df["date"] == pd.to_datetime("2026-05-12").date()) & (df["table"].isin(["table_2", "table_3", "table_4"])),

    (df["date"] == pd.to_datetime("2026-05-13").date()) & (df["table"] == "table_1"),
    (df["date"] == pd.to_datetime("2026-05-13").date()) & (df["table"].isin(["table_2", "table_3", "table_4"])),

    (df["date"] == pd.to_datetime("2026-05-15").date()) & (df["table"].isin(["table_1", "table_2", "table_3", "table_4"]))
]

choices = [1, 0, 1, 0, 1]

df["in_shadow"] = np.select(
    conditions,
    choices,
    default=df["in_shadow"]
)

df = df.drop(columns=["date"])

# Save
df.to_csv(output_path, index=False)