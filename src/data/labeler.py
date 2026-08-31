"""
Labeler: neutral long-format CGM data -> same data plus hypoglycemia labels.
Dataset-agnostic. Runs on the native-resolution neutral file (before resampling),
because episodes must be defined on the finest available series.

Adds three columns:
  in_episode      : True if this reading is part of a qualifying low episode
                    (glucose < LOW_THRESHOLD for >= MIN_DURATION_MIN minutes,
                    not bridging gaps).
  forecast_target : glucose value HORIZON_MIN minutes ahead (for the forecast task).
  alert_label     : True if a qualifying low episode occurs within the next
                    HORIZON_MIN minutes (for the alert task).

Thresholds are in mg/dL. Duration is in minutes and converted to a number of
readings per participant using that participant's own sampling step.
"""

import pandas as pd

LOW_THRESHOLD = 70.0      # mg/dL; 54.0 is the serious tier (too sparse on BrisT1D alone)
MIN_DURATION_MIN = 15     # a low must persist this long to count as an episode
HORIZON_MIN = 60          # look-ahead for the alert and the forecast target


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply labeling per participant and return the concatenated result."""
    pieces = [
        _label_one_participant(g)
        for _, g in df.groupby("participant_id", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def _label_one_participant(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)

    # detect this participant's sampling step (minutes per reading)
    step = g["timestamp"].diff().dropna().mode().iloc[0]
    step_min = step.total_seconds() / 60.0
    readings_needed = max(1, round(MIN_DURATION_MIN / step_min))   # e.g. 1 @15min, 3 @5min
    horizon_steps = round(HORIZON_MIN / step_min)                  # look-ahead in readings

    low = g["glucose"] < LOW_THRESHOLD          # True where below threshold
    # NaN glucose is not "low" and must break a run (can't bridge a gap)
    low = low & g["glucose"].notna()

    # PASS 1 -- mark qualifying episodes: runs of >= readings_needed consecutive lows
    in_episode = _mark_episodes(low.values, readings_needed)
    g["in_episode"] = in_episode

    # forecast target: glucose horizon_steps ahead
    g["forecast_target"] = g["glucose"].shift(-horizon_steps)

    # PASS 2 -- alert label: does a qualifying episode fall within the next horizon?
    # positive if any in_episode reading occurs in rows (i+1 .. i+horizon_steps)
    ep = pd.Series(in_episode)
    future_has_episode = (
        ep.shift(-1).rolling(window=horizon_steps, min_periods=1)
        .max()  # 1 if any episode in the forward window
    )
    # rolling with shift is fiddly; do it explicitly for correctness:
    alert = []
    n = len(g)
    for i in range(n):
        window = in_episode[i + 1 : i + 1 + horizon_steps]
        alert.append(bool(any(window)))
    g["alert_label"] = alert

    return g


def _mark_episodes(low_array, readings_needed):
    """Return a boolean list: True where a reading is inside a run of
    >= readings_needed consecutive lows."""
    n = len(low_array)
    in_ep = [False] * n
    run_start = None
    for i in range(n + 1):
        is_low = low_array[i] if i < n else False
        if is_low:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= readings_needed:
                    for j in range(run_start, i):
                        in_ep[j] = True
                run_start = None
    return in_ep


if __name__ == "__main__":
    # self-test on crafted data
    idx = pd.date_range("2020-01-01 00:00", periods=12, freq="15min")
    g = [120,110,90,68,65,80,120,130,60,120,120,120]
    df = pd.DataFrame({"participant_id": "p01", "timestamp": idx, "glucose": g})
    out = add_labels(df)
    print(out[["timestamp","glucose","in_episode","alert_label","forecast_target"]])