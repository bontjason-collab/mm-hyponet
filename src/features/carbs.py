"""
Carbs-on-board (COB) feature.

Adds a 'cob' column: for each timestamp, total grams of recent meals still being
absorbed, summed over all prior meals via the bilinear (piecewise-parabolic) model.

Absorption parameters (t_peak, DIA) come from an absorption-class lookup. With no
meal-composition text (BrisT1D), every meal uses the 'unknown' default = medium
(t_peak=40, DIA=180 min). Computed per participant over the full series, before
windowing. Same accumulate-then-window pattern as IOB.
"""

import numpy as np
import pandas as pd

# absorption-class table (Loop community-standard times), minutes.
# Only 'medium'/'unknown' is used on BrisT1D (no composition info).
ABSORPTION_CLASSES = {
    "fast":    {"t_peak": 15, "dia": 30},
    "medium":  {"t_peak": 40, "dia": 180},
    "slow":    {"t_peak": 40, "dia": 300},
    "unknown": {"t_peak": 40, "dia": 180},   # = medium
}
DEFAULT_CLASS = "unknown"


def _cob_fraction_curve(t_peak, dia, step_min):
    """Fraction of a meal still on board at k*step_min after eating."""
    n = int(np.ceil(dia / step_min)) + 1
    frac = np.zeros(n)
    r_max = 2.0 / dia          # per-gram rate so absorbed area = 1
    D_rem = dia - t_peak
    for k in range(n):
        t = k * step_min
        if t <= 0:
            absorbed = 0.0
        elif t <= t_peak:
            absorbed = 0.5 * r_max * t * t / t_peak
        elif t <= dia:
            tau = t - t_peak
            absorbed = 0.5 * r_max * t_peak + r_max * tau * (1 - tau / (2 * D_rem))
        else:
            absorbed = 1.0
        frac[k] = max(0.0, 1.0 - absorbed)
    return frac


def add_cob(df, absorption_class=DEFAULT_CLASS):
    """Add a 'cob' column, computed per participant over the full series."""
    params = ABSORPTION_CLASSES[absorption_class]
    tp, dia = params["t_peak"], params["dia"]

    pieces = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        step = g["timestamp"].diff().dropna().mode().iloc[0].total_seconds() / 60.0
        frac = _cob_fraction_curve(tp, dia, step)

        # meals (NaN -> 0: no meal in that interval)
        meals = g["carbs"].fillna(0).to_numpy()
        # COB at each time = sum over past meals of grams * fraction_remaining (causal convolution)
        g["cob"] = np.convolve(meals, frac)[:len(meals)]
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/brist1d/brist1d_features.csv", parse_dates=["timestamp"])
    out = add_cob(df)
    # show around a meal
    meal_rows = out[out["carbs"] > 0].head(1)
    if len(meal_rows):
        idx = meal_rows.index[0]
        print(out.loc[idx-1:idx+8, ["timestamp","carbs","cob"]].to_string())
    print("\ncob stats:", {k: round(v,3) for k,v in out["cob"].describe()[["mean","min","max"]].to_dict().items()})