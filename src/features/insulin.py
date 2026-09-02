"""
Insulin-on-board (IOB) feature.

Adds an 'iob' column to the neutral resampled data: for each timestamp, the total
amount of recent insulin still active, computed from all prior doses via the
OpenAPS/Loop exponential model.

Parameters (tp = time-to-peak, td = Duration of Insulin Action) come from a
per-participant lookup, defaulting to rapid-acting (tp=75, td=300 min) -- correct
for BrisT1D's pump-delivered insulin. Computed over each participant's full
timeline (IOB is a running quantity), BEFORE windowing.
"""

import numpy as np
import pandas as pd

# default rapid-acting parameters; a real lookup keyed on (dataset, participant)
# would override these where known.
DEFAULT_TP = 75.0    # minutes to peak
DEFAULT_TD = 300.0   # minutes duration of insulin action


def _iob_fraction_curve(tp, td, step_min):
    """Fraction of a dose still active at k*step_min after delivery, k=0..td/step."""
    tau = tp * (1 - tp/td) / (1 - 2*tp/td)
    a = 2 * tau / td
    S = 1 / (1 - a + (1 + a) * np.exp(-td/tau))
    n = int(np.ceil(td / step_min)) + 1
    frac = np.zeros(n)
    for k in range(n):
        t = k * step_min
        if t <= 0:
            frac[k] = 1.0
        elif t >= td:
            frac[k] = 0.0
        else:
            frac[k] = 1 - S*(1 - a) * ((t**2/(tau*td*(1-a)) - t/tau - 1)*np.exp(-t/tau) + 1)
    return frac


def add_iob(df, tp=DEFAULT_TP, td=DEFAULT_TD):
    """Add an 'iob' column, computed per participant over the full series."""
    pieces = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        # detect this participant's grid step (minutes)
        step = g["timestamp"].diff().dropna().mode().iloc[0].total_seconds() / 60.0
        frac = _iob_fraction_curve(tp, td, step)

        # insulin doses (NaN -> 0: a gap or no-dose means no insulin delivered)
        doses = g["insulin"].fillna(0).to_numpy()

        # IOB at each time = sum over past doses of dose * fraction_remaining.
        # Efficient: convolve the dose series with the fraction curve (causal).
        iob = np.convolve(doses, frac)[:len(doses)]
        g["iob"] = iob
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/brist1d/brist1d_resampled.csv", parse_dates=["timestamp"])
    out = add_iob(df)
    print(out[["participant_id","timestamp","insulin","iob"]].head(20).to_string())
    print("\niob stats:", out["iob"].describe()[["mean","min","max"]].to_dict())