"""
Resample: neutral labeled data (native resolution) -> common GRID_MINUTES grid.
Aggregates only the columns that exist, so it works for feature-rich datasets
(BrisT1D) and glucose-only ones (DiaData) alike. Timestamp-driven binning.
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

    agg = {}
    agg["glucose"] = r["glucose"].last()                      # level: aligned bin-edge value
    for col in ["insulin", "carbs", "steps", "calories"]:     # amounts: sum
        if col in g.columns:
            agg[col] = r[col].sum(min_count=1)
    if "heart_rate" in g.columns:                             # rate: mean
        agg["heart_rate"] = r["heart_rate"].mean()
    if "activity_type" in g.columns:                          # text: first non-null
        agg["activity_type"] = r["activity_type"].apply(
            lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    for col in ["in_episode", "forecast_target", "alert_label"]:   # labels: bin-edge value
        if col in g.columns:
            agg[col] = r[col].last()
    for col in ["dataset", "split"]:                          # tags: constant per participant
        if col in g.columns:
            agg[col] = r[col].last()

    out = pd.DataFrame(agg)
    out.insert(0, "participant_id", pid)
    return out.reset_index()