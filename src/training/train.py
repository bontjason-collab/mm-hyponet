"""
Model-agnostic training driver for MM-HypoNet.

What varies (set at the top, or later from a config file):
  MODEL_NAME  -- which architecture (tcn now; transformer/lstm later)
  FEATURES    -- which input columns (glucose-only now; +insulin, +carbs, ... later)
  TASK        -- "forecast" (predict glucose value) or "alert" (predict low, later)

What stays fixed: this driver, the windowing, the evaluation. Adding a model means
writing a new architecture file with the same (batch, n_inputs, lookback) -> (batch,)
interface and registering it below. Adding features means extending FEATURES.

Split: participants held out here (VAL_PIDS) are a VALIDATION set, within BrisT1D's
training file. BrisT1D's separate test.csv is the true test set, touched only at the end.
Normalization uses train participants only (no leakage).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# --- model registry: map a name to an architecture class ---
from tcn import TCN
MODEL_REGISTRY = {
    "tcn": TCN,
    # "transformer": Transformer,   # future: same input/output interface
    # "lstm": LSTM,
}

# ============ EXPERIMENT SETTINGS (the ablation knobs) ============
MODEL_NAME = "tcn"
FEATURES   = ["glucose", "iob", "cob", "activity_decay_fast", "activity_decay_medium", "activity_decay_slow"]
TASK       = "forecast"           # "forecast" | "alert"
LOOKBACK   = 12                   # 3 hours at 15-min
VAL_PIDS   = ["p04", "p10"]
EPOCHS     = 15
BATCH      = 256
LR         = 1e-3
SEED       = 42
# =================================================================


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_windows(df, pids, features, task):
    """
    Build (X, y, who) windows.
    X shape: (n_windows, n_features, lookback) -- channels-first for Conv1d.
    y: forecast_target (forecast) or alert_label (alert), aligned to the last input row.
    Windows containing any NaN in the feature inputs or the target are skipped.
    """
    X, y, who = [], [], []
    target_col = "forecast_target" if task == "forecast" else "alert_label"
    sub = df[df["participant_id"].isin(pids)]
    for pid, g in sub.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feats = g[features].to_numpy(dtype=np.float32)   # (rows, n_features)
        tgt = g[target_col].to_numpy(dtype=np.float32)
        for i in range(LOOKBACK, len(g)):
            w = feats[i - LOOKBACK:i, :]                  # (lookback, n_features)
            t = tgt[i - 1]
            if np.isnan(w).any() or np.isnan(t):
                continue
            X.append(w.T)                                 # -> (n_features, lookback)
            y.append(t); who.append(pid)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(who))


def normalize(Xtr, Xval):
    """Per-feature standardization using TRAIN stats only. X: (n, n_features, lookback)."""
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xval - mu) / sd


def metrics(pid, y_true, y_pred, task):
    if task == "forecast":
        err = y_pred - y_true
        return {"participant_id": pid, "n": len(y_true),
                "MAE_mgdl": round(float(np.mean(np.abs(err))), 2),
                "RMSE_mgdl": round(float(np.sqrt(np.mean(err ** 2))), 2)}
    else:
        # alert: y_pred is a probability; report at 0.5 for now (thresholds tuned later)
        pred_bin = (y_pred >= 0.5).astype(int)
        tp = int(((pred_bin == 1) & (y_true == 1)).sum())
        fp = int(((pred_bin == 1) & (y_true == 0)).sum())
        fn = int(((pred_bin == 0) & (y_true == 1)).sum())
        tn = int(((pred_bin == 0) & (y_true == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        far  = fp / (fp + tn) if (fp + tn) else 0.0
        return {"participant_id": pid, "n": len(y_true),
                "sensitivity": round(sens, 3), "false_alarm_rate": round(far, 3)}


def run(csv_path):
    set_seed(SEED)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    all_pids = list(df["participant_id"].unique())
    train_pids = [p for p in all_pids if p not in VAL_PIDS]

    Xtr, ytr, _ = build_windows(df, train_pids, FEATURES, TASK)
    Xval, yval, who_val = build_windows(df, VAL_PIDS, FEATURES, TASK)
    Xtr_n, Xval_n = normalize(Xtr, Xval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ModelClass = MODEL_REGISTRY[MODEL_NAME]
    model = ModelClass(n_inputs=len(FEATURES), channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss() if TASK == "forecast" else torch.nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.tensor(Xtr_n), torch.tensor(ytr))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)

        # validation each epoch (watch for overfitting)
        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(Xval_n).to(device)).cpu().numpy()
        if TASK == "forecast":
            val_score = np.mean(np.abs(vp - yval))
            print(f"epoch {ep+1}/{EPOCHS}  train {tot/len(tr_ds):.3f}  val MAE {val_score:.2f}")
        else:
            vp_prob = 1 / (1 + np.exp(-vp))   # sigmoid on logits
            val_score = np.mean(np.abs(vp_prob - yval))
            print(f"epoch {ep+1}/{EPOCHS}  train {tot/len(tr_ds):.3f}  val loss-proxy {val_score:.3f}")

    # final per-participant evaluation on validation participants
    model.eval()
    with torch.no_grad():
        vp = model(torch.tensor(Xval_n).to(device)).cpu().numpy()
    if TASK == "alert":
        vp = 1 / (1 + np.exp(-vp))            # probabilities for metric

    rows = [metrics(pid, yval[who_val == pid], vp[who_val == pid], TASK) for pid in VAL_PIDS]
    rows.append(metrics("ALL_VAL", yval, vp, TASK))
    result = pd.DataFrame(rows)
    result["experiment"] = f"{MODEL_NAME}_{'+'.join(FEATURES)}_{TASK}"
    return result


if __name__ == "__main__":
    res = run("data/processed/brist1d/brist1d_features.csv")
    print(res.to_string(index=False))