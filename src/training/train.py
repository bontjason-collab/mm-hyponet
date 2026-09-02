"""
Model-agnostic training driver. Settings are passed to run() as arguments, so the
file is a fixed engine -- you never edit it to change an experiment. The notebook
cell picks features/task per run.

  task="forecast": MAE loss, MAE/RMSE metrics.
  task="alert":    focal loss + balanced sampling, sensitivity/false-alarm metrics.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from tcn import TCN
MODEL_REGISTRY = {"tcn": TCN}

LOOKBACK = 12
VAL_PIDS = ["p04", "p10"]
SEED = 42
FOCAL_GAMMA = 2.0


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_windows(df, pids, features, task):
    X, y, who = [], [], []
    target_col = "forecast_target" if task == "forecast" else "alert_label"
    sub = df[df["participant_id"].isin(pids)]
    for pid, g in sub.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feats = g[features].to_numpy(dtype=np.float32)
        tgt = g[target_col].to_numpy(dtype=np.float32)
        for i in range(LOOKBACK, len(g)):
            w = feats[i - LOOKBACK:i, :]
            t = tgt[i - 1]
            if np.isnan(w).any() or np.isnan(t):
                continue
            X.append(w.T); y.append(t); who.append(pid)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(who))


def normalize(Xtr, Xval):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xval - mu) / sd


def focal_loss(logits, targets, gamma=FOCAL_GAMMA):
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * ce).mean()


def forecast_metrics(pid, y_true, y_pred):
    err = y_pred - y_true
    return {"participant_id": pid, "n": len(y_true),
            "MAE_mgdl": round(float(np.mean(np.abs(err))), 2),
            "RMSE_mgdl": round(float(np.sqrt(np.mean(err ** 2))), 2)}


def alert_metrics(pid, y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    far  = fp / (fp + tn) if (fp + tn) else 0.0
    return {"participant_id": pid, "n": len(y_true), "n_pos": int(y_true.sum()),
            "sensitivity": round(sens, 3), "false_alarm_rate": round(far, 3),
            "tp": tp, "fp": fp, "fn": fn}


def run(csv_path, features, task, epochs=15, model_name="tcn", batch=256, lr=1e-3):
    set_seed(SEED)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    all_pids = list(df["participant_id"].unique())
    train_pids = [p for p in all_pids if p not in VAL_PIDS]

    Xtr, ytr, _ = build_windows(df, train_pids, features, task)
    Xval, yval, who_val = build_windows(df, VAL_PIDS, features, task)
    Xtr_n, Xval_n = normalize(Xtr, Xval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODEL_REGISTRY[model_name](n_inputs=len(features), channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_ds = TensorDataset(torch.tensor(Xtr_n), torch.tensor(ytr))

    if task == "alert":
        pos = ytr.sum(); neg = len(ytr) - pos
        sample_w = np.where(ytr == 1, 1.0 / max(pos, 1), 1.0 / max(neg, 1))
        sampler = WeightedRandomSampler(sample_w, num_samples=len(ytr), replacement=True)
        tr_dl = DataLoader(tr_ds, batch_size=batch, sampler=sampler)
        loss_fn = focal_loss
    else:
        tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
        loss_fn = torch.nn.L1Loss()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(Xval_n).to(device)).cpu().numpy()
        if task == "forecast":
            print(f"epoch {ep+1}/{epochs}  train {tot/len(tr_ds):.3f}  val MAE {np.mean(np.abs(vp - yval)):.2f}")
        else:
            prob = 1 / (1 + np.exp(-vp))
            m = alert_metrics("val", yval, prob)
            print(f"epoch {ep+1}/{epochs}  train {tot/len(tr_ds):.4f}  val sens {m['sensitivity']:.3f}  far {m['false_alarm_rate']:.3f}")

    model.eval()
    with torch.no_grad():
        vp = model(torch.tensor(Xval_n).to(device)).cpu().numpy()
    if task == "alert":
        vp = 1 / (1 + np.exp(-vp))
        rows = [alert_metrics(pid, yval[who_val == pid], vp[who_val == pid]) for pid in VAL_PIDS]
        rows.append(alert_metrics("ALL_VAL", yval, vp))
    else:
        rows = [forecast_metrics(pid, yval[who_val == pid], vp[who_val == pid]) for pid in VAL_PIDS]
        rows.append(forecast_metrics("ALL_VAL", yval, vp))

    result = pd.DataFrame(rows)
    result["experiment"] = f"{model_name}_{'+'.join(features)}_{task}"
    return result