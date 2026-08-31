"""
Adapter: BrisT1D raw competition CSV -> neutral long format.

Takes each row's own-timestamp ('-0:00') columns, in row order per participant,
to reconstruct the underlying series. Rows are ~15 min apart with occasional
gaps. Timestamps are rebuilt from the clock-only 'time' field, adding a calendar
day each time the clock rolls past midnight, then placed on a strict 15-min grid
so gaps become rows with a real timestamp + participant_id but NaN measurements.

Neutral output columns:
  participant_id, timestamp, glucose, insulin, carbs, heart_rate, steps, calories
Glucose converted mmol/L -> mg/dL.
"""

import pandas as pd

MMOL_TO_MGDL = 18.0182
STEP = pd.Timedelta(minutes=15)


def brist1d_to_long(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path, low_memory=False)

    now_cols = {
        "bg-0:00":      "glucose",
        "insulin-0:00": "insulin",
        "carbs-0:00":   "carbs",
        "hr-0:00":      "heart_rate",
        "steps-0:00":   "steps",
        "cals-0:00":    "calories",
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
        "insulin", "carbs", "heart_rate", "steps", "calories",
    ]]


def _rebuild_one_participant(pid: str, g: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monotonic datetime from clock-only 'time', rolling to the next
    calendar day whenever the clock goes backwards (past-midnight), then
    reindex onto a strict 15-min grid so gaps are explicit NaN rows.
    """
    g = g.reset_index(drop=True)
    tod = pd.to_timedelta(g["time"])          # time-of-day as duration

    base = pd.Timestamp("2020-01-01")
    day_offset = pd.Timedelta(0)
    timestamps = []
    prev_tod = None
    for cur_tod in tod:
        if prev_tod is not None and cur_tod < prev_tod:
            # clock went backwards -> we crossed midnight, add a day
            day_offset += pd.Timedelta(days=1)
        timestamps.append(base + day_offset + cur_tod)
        prev_tod = cur_tod

    g = g.assign(timestamp=timestamps).drop(columns=["time"])

    # de-duplicate any exact timestamp collisions before reindexing
    g = g.drop_duplicates(subset="timestamp", keep="first")

    # strict 15-min grid; gaps become rows with NaN measurements
    full = pd.date_range(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], freq=STEP)
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