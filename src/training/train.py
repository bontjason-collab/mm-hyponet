"""
Model-agnostic training driver for MM-HypoNet.

run() args: csv_path, features, task ("forecast"|"alert"), epochs, model_name, batch, lr.

Split selection:
  - If the data has a 'split' column (DiaData: train/val/holdout), use it, and
    evaluate the holdout PER COHORT (the cross-cohort robustness test).
  - Else fall back to VAL_PIDS (BrisT1D: hold out named participants).

Alert task: focal loss + balanced sampling; reports sensitivity/false-alarm/AUC,
and per-epoch validation AUC. No early stopping (fixed epochs, fair across models).
Returns (metrics_table, extras) where extras carries probabilities/labels for ROC.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from tcn import TCN
MODEL_REGISTRY = {"tcn": TCN}

LOOKBACK = 12
VAL_PIDS = ["p04", "p10"]      # fallback when no 'split' column (BrisT1D)
SEED = 42
FOCAL_GAMMA = 2.0


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_windows(df, features, task):
    """Build windows from a dataframe (already filtered to the desired rows)."""
    X, y, who, coh = [], [], [], []
    target_col = "forecast_target" if task == "forecast" else "alert_label"
    has_cohort = "dataset" in df.columns
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feats = g[features].to_numpy(dtype=np.float32)
        tgt = g[target_col].to_numpy(dtype=np.float32)
        cohort = g["dataset"].iloc[0] if has_cohort else "NA"
        for i in range(LOOKBACK, len(g)):
            w = feats[i - LOOKBACK:i, :]
            t = tgt[i - 1]
            if np.isnan(w).any() or np.isnan(t):
                continue
            X.append(w.T); y.append(t); who.append(pid); coh.append(cohort)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(who), np.array(coh))


def normalize(Xtr, *others, mu=None, sd=None):
    if mu is None:
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others] + [mu, sd]


def focal_loss(logits, targets, gamma=FOCAL_GAMMA):
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * ce).mean()


def forecast_metrics(name, y_true, y_pred):
    err = y_pred - y_true
    return {"group": name, "n": len(y_true),
            "MAE_mgdl": round(float(np.mean(np.abs(err))), 2),
            "RMSE_mgdl": round(float(np.sqrt(np.mean(err ** 2))), 2)}


def alert_metrics(name, y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    far  = fp / (fp + tn) if (fp + tn) else 0.0
    auc = (roc_auc_score(y_true, y_prob)
           if (0 < y_true.sum() < len(y_true)) else float("nan"))
    return {"group": name, "n": len(y_true), "n_pos": int(y_true.sum()),
            "sensitivity@0.5": round(sens, 3), "false_alarm@0.5": round(far, 3),
            "AUC": round(auc, 4)}


def run(csv_path, features, task, epochs=15, model_name="tcn", batch=256, lr=1e-3):
    set_seed(SEED)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # ---- split selection ----
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        val_df   = df[df["split"] == "val"]
        holdout_df = df[df["split"] == "holdout"]
    else:
        train_df = df[~df["participant_id"].isin(VAL_PIDS)]
        val_df   = df[df["participant_id"].isin(VAL_PIDS)]
        holdout_df = df.iloc[0:0]   # empty

    Xtr, ytr, _, _ = build_windows(train_df, features, task)
    Xval, yval, who_val, coh_val = build_windows(val_df, features, task)
    Xtr_n, Xval_n, mu, sd = normalize(Xtr, Xval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MODEL_REGISTRY[model_name](n_inputs=len(features), channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_ds = TensorDataset(torch.tensor(Xtr_n), torch.tensor(ytr))
    if task == "alert":
        pos = ytr.sum(); neg = len(ytr) - pos
        w = np.where(ytr == 1, 1.0 / max(pos, 1), 1.0 / max(neg, 1))
        sampler = WeightedRandomSampler(w, num_samples=len(ytr), replacement=True)
        tr_dl = DataLoader(tr_ds, batch_size=batch, sampler=sampler)
        loss_fn = focal_loss
    else:
        tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
        loss_fn = torch.nn.L1Loss()

    Xval_t = torch.tensor(Xval_n).to(device)
    auc_history = []

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
            auc = roc_auc_score(yval, prob) if 0 < yval.sum() < len(yval) else float("nan")
            auc_history.append(auc)
            m = alert_metrics("val", yval, prob)
            print(f"epoch {ep+1}/{epochs}  train {tot/len(tr_ds):.4f}  val AUC {auc:.4f}  "
                  f"sens {m['sensitivity@0.5']:.3f}  far {m['false_alarm@0.5']:.3f}")

    # ---- final evaluation: val (overall + per cohort) and holdout per cohort ----
    def eval_group(Xg_n, yg, who_g, coh_g, label):
        rows = []
        with torch.no_grad():
            og = model(torch.tensor(Xg_n).to(device)).cpu().numpy()
        if task == "alert":
            pg = 1 / (1 + np.exp(-og))
            rows.append(alert_metrics(f"{label}_ALL", yg, pg))
            for c in np.unique(coh_g):
                mask = coh_g == c
                rows.append(alert_metrics(f"{label}_{c}", yg[mask], pg[mask]))
            return rows, pg
        else:
            rows.append(forecast_metrics(f"{label}_ALL", yg, og))
            for c in np.unique(coh_g):
                mask = coh_g == c
                rows.append(forecast_metrics(f"{label}_{c}", yg[mask], og[mask]))
            return rows, og

    all_rows = []
    val_rows, val_pred = eval_group(Xval_n, yval, who_val, coh_val, "val")
    all_rows += val_rows

    holdout_extras = None
    if len(holdout_df) > 0:
        Xho, yho, who_ho, coh_ho = build_windows(holdout_df, features, task)
        Xho_n, _, _ = normalize(Xho, mu=mu, sd=sd)   # normalize with TRAIN stats
        ho_rows, ho_pred = eval_group(Xho_n, yho, who_ho, coh_ho, "holdout")
        all_rows += ho_rows
        holdout_extras = {"prob": ho_pred, "label": yho, "cohort": coh_ho}

    result = pd.DataFrame(all_rows)
    result["experiment"] = f"{model_name}_{'+'.join(features)}_{task}"

    extras = {"val_prob": val_pred, "val_label": yval, "val_cohort": coh_val,
              "auc_history": auc_history, "holdout": holdout_extras,
              "experiment": f"{model_name}_{'+'.join(features)}_{task}"}
    return result, extras