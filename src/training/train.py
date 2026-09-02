"""
Model-agnostic training driver. Settings passed to run() as arguments.

  task="forecast": MAE loss; returns (metrics_table, extras).
  task="alert":    focal loss + balanced sampling; returns (metrics_table, extras).

No early stopping -- trains a fixed number of epochs (same for every feature set,
so comparisons stay fair). For the alert task, computes validation ROC-AUC each
epoch (threshold-free and not dominated by the majority class), and returns:
  - per-epoch AUC history (to see convergence / overfitting)
  - final validation probabilities + labels (to plot the ROC curve)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

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
    auc  = roc_auc_score(y_true, y_prob) if (y_true.sum() > 0 and y_true.sum() < len(y_true)) else float("nan")
    return {"participant_id": pid, "n": len(y_true), "n_pos": int(y_true.sum()),
            "sensitivity@0.5": round(sens, 3), "false_alarm@0.5": round(far, 3),
            "AUC": round(auc, 4)}


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

    Xval_t = torch.tensor(Xval_n).to(device)
    auc_history = []      # per-epoch validation AUC (alert task)

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
            out = model(Xval_t).cpu().numpy()

        if task == "forecast":
            print(f"epoch {ep+1}/{epochs}  train {tot/len(tr_ds):.3f}  val MAE {np.mean(np.abs(out - yval)):.2f}")
        else:
            prob = 1 / (1 + np.exp(-out))
            auc = roc_auc_score(yval, prob)
            auc_history.append(auc)
            m = alert_metrics("val", yval, prob)
            print(f"epoch {ep+1}/{epochs}  train {tot/len(tr_ds):.4f}  "
                  f"val AUC {auc:.4f}  sens@0.5 {m['sensitivity@0.5']:.3f}  far@0.5 {m['false_alarm@0.5']:.3f}")

    # ---- final evaluation (epoch-15 model, no early stopping) ----
    model.eval()
    with torch.no_grad():
        out = model(Xval_t).cpu().numpy()

    if task == "alert":
        prob = 1 / (1 + np.exp(-out))
        rows = [alert_metrics(pid, yval[who_val == pid], prob[who_val == pid]) for pid in VAL_PIDS]
        rows.append(alert_metrics("ALL_VAL", yval, prob))
        result = pd.DataFrame(rows)
        result["experiment"] = f"{model_name}_{'+'.join(features)}_{task}"
        extras = {"prob": prob, "label": yval, "who": who_val,
                  "auc_history": auc_history,
                  "experiment": f"{model_name}_{'+'.join(features)}_{task}"}
        return result, extras
    else:
        rows = [forecast_metrics(pid, yval[who_val == pid], out[who_val == pid]) for pid in VAL_PIDS]
        rows.append(forecast_metrics("ALL_VAL", yval, out))
        result = pd.DataFrame(rows)
        result["experiment"] = f"{model_name}_{'+'.join(features)}_{task}"
        extras = {"pred": out, "label": yval, "who": who_val,
                  "experiment": f"{model_name}_{'+'.join(features)}_{task}"}
        return result, extras