"""
Persistence baseline for glucose forecasting.

The baseline to beat: "glucose 60 min from now = glucose right now."
Because glucose moves slowly, this is a strong predictor and the honest floor.
Any real model must beat it clearly, or it has learned nothing useful.

Evaluated per participant (never pooled into one number that hides a bad case),
using MAE and RMSE in mg/dL.
"""

import numpy as np
import pandas as pd


def evaluate_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each participant, predict forecast_target with the current glucose,
    and report MAE and RMSE. Returns one row per participant plus an 'ALL' row.
    """
    rows = []
    for pid, g in df.groupby("participant_id", sort=False):
        valid = g["glucose"].notna() & g["forecast_target"].notna()
        y_true = g.loc[valid, "forecast_target"].to_numpy()
        y_pred = g.loc[valid, "glucose"].to_numpy()   # persistence: predict "now"
        rows.append(_metrics(pid, y_true, y_pred))

    # overall (pooled across all valid rows)
    valid_all = df["glucose"].notna() & df["forecast_target"].notna()
    rows.append(_metrics(
        "ALL",
        df.loc[valid_all, "forecast_target"].to_numpy(),
        df.loc[valid_all, "glucose"].to_numpy(),
    ))

    return pd.DataFrame(rows)


def _metrics(pid, y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    return {"participant_id": pid, "n": len(y_true),
            "MAE_mgdl": round(mae, 2), "RMSE_mgdl": round(rmse, 2)}


if __name__ == "__main__":
    df = pd.read_csv(
        "data/processed/brist1d/brist1d_resampled.csv",
        parse_dates=["timestamp"],
    )
    result = evaluate_persistence(df)
    print(result.to_string(index=False))