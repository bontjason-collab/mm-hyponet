"""
Adapter: BrisT1D raw competition CSV -> neutral long format.

BrisT1D packs a 6-hour history into 500+ overlapping columns per row. The '-0:00'
columns hold each row's own-timestamp reading, so taking just those, in row order
per participant, reconstructs the underlying series.

IMPORTANT: BrisT1D is NOT uniformly sampled. Some participants are recorded every
5 minutes, others every 15 minutes. This adapter detects each participant's native
spacing and places their readings on that participant's own grid. It does NOT
standardize resolution across participants -- that is a later, dataset-agnostic
resample step -- so the adapter stays honest (native resolution, no fabrication).

Self-declared activity text is carried through RAW and unmodified (activity_type).
Normalizing free-text activity/meal labels into a fixed category set is a separate,
dataset-agnostic converter run later, so the adapter never bakes in dataset-specific
cleaning decisions.

Timestamps are rebuilt from the clock-only 'time' field (HH:MM:SS), adding a
calendar day whenever the clock rolls past midnight. Gaps become rows with a real
timestamp + participant_id but NaN/empty measurements.

Neutral output columns:
  participant_id, timestamp, glucose, insulin, carbs,
  heart_rate, steps, calories, activity_type
Glucose is converted mmol/L -> mg/dL. activity_type is raw text ('' when none).
"""

import pandas as pd

MMOL_TO_MGDL = 18.0182


def brist1d_to_long(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path, low_memory=False)

    # The "current moment" value of each signal is its '-0:00' lag column.
    now_cols = {
        "bg-0:00":       "glucose",
        "insulin-0:00":  "insulin",
        "carbs-0:00":    "carbs",
        "hr-0:00":       "heart_rate",
        "steps-0:00":    "steps",
        "cals-0:00":     "calories",
        "activity-0:00": "activity_type",   # raw self-declared text, preserved as-is
    }
    keep = ["p_num", "time"] + list(now_cols.keys())
    out = df[keep].copy().rename(columns=now_cols)
    out = out.rename(columns={"p_num": "participant_id"})
    out["glucose"] = out["glucose"] * MMOL_TO_MGDL

    pieces = [
        _rebuild_one_participant(pid, g)
        for pid, g in out.groupby("participant_id", sort=False)
    ]
    result = pd.concat(pieces, ignore_index=True)
    return result[[
        "participant_id", "timestamp", "glucose",
        "insulin", "carbs", "heart_rate", "steps", "calories", "activity_type",
    ]]


def _rebuild_one_participant(pid: str, g: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monotonic datetime from clock-only 'time' (rolling to the next
    calendar day past midnight), detect this participant's native sampling
    step from the data, then reindex onto that step so gaps are explicit rows.
    """
    g = g.reset_index(drop=True)
    tod = pd.to_timedelta(g["time"])          # time-of-day as duration

    base = pd.Timestamp("2020-01-01")
    day_offset = pd.Timedelta(0)
    timestamps = []
    prev_tod = None
    for cur_tod in tod:
        if prev_tod is not None and cur_tod < prev_tod:
            day_offset += pd.Timedelta(days=1)   # crossed midnight
        timestamps.append(base + day_offset + cur_tod)
        prev_tod = cur_tod

    g = g.assign(timestamp=timestamps).drop(columns=["time"])
    g = g.drop_duplicates(subset="timestamp", keep="first").sort_values("timestamp")

    # detect THIS participant's native spacing (5 or 15 min) from the data
    diffs = g["timestamp"].diff().dropna()
    step = diffs.mode().iloc[0]

    # reindex onto that participant's own grid; gaps become NaN/empty rows
    full = pd.date_range(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], freq=step)
    g = g.set_index("timestamp").reindex(full)
    g["participant_id"] = pid
    g.index.name = "timestamp"
    return g.reset_index()


if __name__ == "__main__":
    long_df = brist1d_to_long("data/raw/brist1d/train.csv")
    print(long_df.head(20))
    print("\nrows:", len(long_df))
    print("participants:", long_df["participant_id"].nunique())
    print("gap rows (glucose NaN):", long_df["glucose"].isna().sum())
    print("activity_type non-empty:", long_df["activity_type"].notna().sum())
    print("distinct activities:", long_df["activity_type"].dropna().nunique())