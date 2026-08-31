"""
Adapter: BrisT1D raw competition CSV -> neutral long format.

BrisT1D packs a 6-hour history into 500+ overlapping columns per row. The '-0:00'
columns hold each row's own-timestamp reading, so taking just those, in row order
per participant, reconstructs the underlying series. Rows are ~15 min apart BUT
gaps occur (skipped readings). We rebuild the timeline from the clock-only 'time'
field (handling midnight rollovers) and place every reading on a strict 15-minute
grid. Gap slots are kept as rows with a real timestamp + participant_id but NaN
measurements, so downstream masking and labeling see them as genuinely missing.

Neutral output columns:
  participant_id, timestamp, glucose, insulin, carbs, heart_rate, steps, calories
Glucose is converted mmol/L -> mg/dL.
"""

import pandas as pd

MMOL_TO_MGDL = 18.0182
STEP = pd.Timedelta(minutes=15)


def brist1d_to_long(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # The "current moment" value of each signal is its '-0:00' lag column.
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
    Reconstruct a monotonic 15-min timeline from the clock-only 'time' field,
    handling midnight rollovers, then place readings on a strict 15-min grid so
    gaps become explicit rows with NaN measurements.
    """
    g = g.reset_index(drop=True)
    clock = pd.to_timedelta(g["time"])

    base_date = pd.Timestamp("2020-01-01")
    timestamps = [base_date + clock.iloc[0]]
    prev = timestamps[0]
    for i in range(1, len(g)):
        target_tod = clock.iloc[i]                    # desired time-of-day
        t = prev + STEP
        steps = 0
        # advance in 15-min steps until time-of-day matches (spans gaps + midnight);
        # cap the search so a malformed row can't loop forever (7 days)
        while (t - t.normalize()) != target_tod and steps < 4 * 24 * 7:
            t += STEP
            steps += 1
        timestamps.append(t)
        prev = t

    g = g.assign(timestamp=timestamps).drop(columns=["time"])

    # Strict continuous 15-min grid: every slot gets a timestamp row.
    # Gap slots have a real timestamp but NaN measurements.
    full = pd.date_range(g["timestamp"].iloc[0], g["timestamp"].iloc[-1], freq=STEP)
    g = g.set_index("timestamp").reindex(full)

    # Fill identifiers so only the measurement columns are blank on gap rows.
    g["participant_id"] = pid
    g.index.name = "timestamp"
    return g.reset_index()


if __name__ == "__main__":
    # quick manual check
    long_df = brist1d_to_long("data/raw/brist1d/train.csv")
    print(long_df.head(20))
    print("\nrows:", len(long_df))
    print("participants:", long_df["participant_id"].nunique())
    print("gap rows (glucose NaN):", long_df["glucose"].isna().sum())