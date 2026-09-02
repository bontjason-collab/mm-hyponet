"""
Activity features:
  1. Intensity decay -- three exponential-decay columns on calories (fast/medium/slow),
     capturing recent exertion at different timescales (the always-on backbone).
  2. Activity type   -- self-declared text mapped to aerobic/anaerobic/mixed via a
     lookup table, with a confidence column. Sparse on BrisT1D (~1.5% of rows), so
     mostly 'unknown'; the seam exists for datasets with richer activity labels.

Calories NaN -> 0 for the decay (no recorded activity = no exertion; a known
limitation, since 'watch not worn' and 'resting' both become 0). Computed per
participant over the full series, before windowing.
"""

import numpy as np
import pandas as pd

# --- decay timescales (per the spec): fade per 15-min step ---
# fast ~1h, medium ~4h, slow ~24h. Stored as the multiplicative fade each step.
DECAY_RATES = {
    "activity_decay_fast":   0.90,    # ~1 hour memory
    "activity_decay_medium": 0.98,    # ~4 hours
    "activity_decay_slow":   0.997,   # ~24 hours
}

# --- activity type lookup: raw text -> category ---
ACTIVITY_TYPE_MAP = {
    # aerobic
    "walk": "aerobic", "walking": "aerobic", "run": "aerobic", "running": "aerobic",
    "hike": "aerobic", "swim": "aerobic", "swimming": "aerobic",
    "outdoor bike": "aerobic", "bike": "aerobic", "spinning": "aerobic",
    "aerobic workout": "aerobic", "dancing": "aerobic", "zumba": "aerobic",
    "yoga": "aerobic",
    # anaerobic
    "weights": "anaerobic", "strength training": "anaerobic", "hiit": "anaerobic",
    "indoor climbing": "anaerobic", "stairclimber": "anaerobic",
    # mixed
    "sport": "mixed", "workout": "mixed", "tennis": "mixed",
}


def _decay_series(values, rate):
    """Running decayed sum: level[t] = rate*level[t-1] + values[t]."""
    out = np.empty(len(values))
    acc = 0.0
    for i, v in enumerate(values):
        acc = rate * acc + v
        out[i] = acc
    return out


def add_activity(df):
    pieces = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        # --- intensity decay on calories (NaN -> 0) ---
        cals = g["calories"].fillna(0).to_numpy(dtype=float)
        for col, rate in DECAY_RATES.items():
            g[col] = _decay_series(cals, rate)

        # --- activity type + confidence ---
        raw = g["activity_type"].astype("string")
        norm = raw.str.strip().str.lower()
        g["activity_type_cat"] = norm.map(ACTIVITY_TYPE_MAP).fillna("unknown")
        # deterministic lookup: confidence 1.0 where a known label mapped, else 0
        g["activity_type_confidence"] = norm.map(
            lambda x: 1.0 if x in ACTIVITY_TYPE_MAP else 0.0
        )

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/brist1d/brist1d_features.csv", parse_dates=["timestamp"])
    out = add_activity(df)
    print("new columns:", [c for c in out.columns if "activity_decay" in c or "activity_type_c" in c])
    print("\nactivity_type_cat counts:")
    print(out["activity_type_cat"].value_counts())
    print("\ndecay stats:")
    print(out[list(DECAY_RATES.keys())].describe().loc[["mean","min","max"]])