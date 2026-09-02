"""
Resample: neutral labeled data (native resolution) -> common GRID_MINUTES grid.

Timestamp-driven: bins by actual clock time using pandas resample, so it never
assumes a fixed number of readings per bin. Works for 5-min, 15-min, or any
irregular spacing. Aggregation is signal-specific:

  glucose      -> aligned value at the bin edge (the reading at that time), matching
                  how a real 15-min monitor samples a level (last reading in the bin).
  insulin      -> SUM  (per-interval units delivered: basal drip + boluses)
  carbs        -> SUM  (per-interval grams)
  steps        -> SUM  (per-interval step count)
  calories     -> SUM  (per-interval burn)
  heart_rate   -> MEAN (already an average rate)
  activity_type-> first non-null string in the bin (keep any declared activity)

Labels (in_episode, forecast_target, alert_label) are carried at the bin edge.
They were built on the native series (per the spec: label before downsampling).
Re-labeling on the resampled grid is a separate, optional step.

A bin with no glucose reading stays a gap (NaN glucose) -- missingness preserved,
never invented. Amounts use min_count=1 so an all-NaN bin stays NaN, not a false 0.
"""

import pandas as pd

GRID_MINUTES = 15


def resample_to_grid(df: pd.DataFrame, grid_minutes: int = GRID_MINUTES) -> pd.DataFrame:
    freq = f"{grid_minutes}min"
    pieces = [
        _resample_one(pid, g, freq)
        for pid, g in df.groupby("participant_id", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def _resample_one(pid: str, g: pd.DataFrame, freq: str) -> pd.DataFrame:
    g = g.sort_values("timestamp").set_index("timestamp")

    r = g.resample(freq, label="right", closed="right")

    out = pd.DataFrame({
        # glucose: aligned value at the bin edge = last real reading in the bin
        "glucose":       r["glucose"].last(),
        # amounts per interval -> sum (min_count=1 so an all-NaN bin stays NaN)
        "insulin":       r["insulin"].sum(min_count=1),
        "carbs":         r["carbs"].sum(min_count=1),
        "steps":         r["steps"].sum(min_count=1),
        "calories":      r["calories"].sum(min_count=1),
        # rate -> mean
        "heart_rate":    r["heart_rate"].mean(),
        # text -> first non-null in the bin
        "activity_type": r["activity_type"].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else None
        ),
        # labels carried at the bin edge (built on native series)
        "in_episode":      r["in_episode"].last(),
        "forecast_target": r["forecast_target"].last(),
        "alert_label":     r["alert_label"].last(),
    })

    out.insert(0, "participant_id", pid)
    out = out.reset_index()
    return out


if __name__ == "__main__":
    df = pd.read_csv("data/processed/brist1d/brist1d_labeled.csv", parse_dates=["timestamp"])
    res = resample_to_grid(df)

    print("rows before:", len(df), " after:", len(res))
    for pid, g in res.groupby("participant_id", sort=False):
        step = g["timestamp"].diff().dropna().mode().iloc[0]
        print(f"  {pid}: grid step={step}, rows={len(g)}")