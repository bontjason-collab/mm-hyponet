"""
Slope-extrapolation baseline for hypoglycemia alerting (Dexcom-style trend method).

For each window, fit a least-squares slope over the last W glucose readings,
project 60 minutes ahead, and score by how far below the low threshold the
projection lands. Higher score = more urgently heading low. The score is
continuous, so it yields a full ROC/PR curve like the learned model.

Sweeps window length W from 2 points (30 min) to 12 points (3 h) and reports
each, so the fairest (best) trend baseline can be compared against the TCN.

Evaluated on the SAME windows/labels the TCN uses (build_windows), so the
comparison is apples-to-apples.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

LOOKBACK = 12          # 3 hours at 15-min (must match the TCN's window)
HORIZON_MIN = 60
STEP_MIN = 15
LOW_THRESHOLD = 70.0


def _windows(df, split_value=None):
    """Build (glucose_window, label, cohort) from resampled labeled data.
    Uses the same gap-skipping rule as the TCN's build_windows."""
    if split_value is not None and "split" in df.columns:
        df = df[df["split"] == split_value]
    Xg, y, coh = [], [], []
    has_cohort = "dataset" in df.columns
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        glu = g["glucose"].to_numpy(dtype=float)
        lab = g["alert_label"].to_numpy(dtype=float)
        cohort = g["dataset"].iloc[0] if has_cohort else "NA"
        for i in range(LOOKBACK, len(g)):
            w = glu[i - LOOKBACK:i]
            t = lab[i - 1]
            if np.isnan(w).any() or np.isnan(t):
                continue
            Xg.append(w); y.append(t); coh.append(cohort)
    return np.array(Xg), np.array(y), np.array(coh)


def _slope_score(window_glucose, W):
    """Score = LOW - projected_glucose_60min, using a least-squares slope
    over the last W points of the 12-point window."""
    sub = window_glucose[-W:]
    t = np.arange(W) * STEP_MIN
    # vectorized least-squares slope for each row
    t_mean = t.mean()
    t_c = t - t_mean
    denom = (t_c ** 2).sum()
    sub_mean = sub.mean(axis=1, keepdims=True)
    slope = ((t_c * (sub - sub_mean)).sum(axis=1)) / denom   # mg/dL per min, per window
    current = sub[:, -1]
    predicted = current + slope * HORIZON_MIN
    return LOW_THRESHOLD - predicted     # higher = more heading-low


def evaluate(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    split_val = "val" if "split" in df.columns else None
    Xg, y, coh = _windows(df, split_value=split_val)
    print(f"windows: {len(Xg)}, positives: {int(y.sum())} ({100*y.mean():.2f}%)")

    rows = []
    for W in [2, 3, 4, 6, 8, 12]:
        score = _slope_score(Xg, W)
        auc = roc_auc_score(y, score)
        prauc = average_precision_score(y, score)
        rows.append({"window_pts": W, "window_min": W * STEP_MIN,
                     "ROC_AUC": round(auc, 4), "PR_AUC": round(prauc, 4)})
    return pd.DataFrame(rows), (Xg, y, coh)


if __name__ == "__main__":
    res, _ = evaluate("data/processed/diadata/diadata_resampled.csv")
    print(res.to_string(index=False))