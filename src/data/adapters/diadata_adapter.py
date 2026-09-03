"""
Adapter: DiaData merged CSV -> neutral long format, subsampled and split-tagged.

DiaData (maindatabase.csv, ~135M rows, 5.6GB) is glucose-only:
  columns ts, PtID, GlucoseCGM (mg/dL), Database.

This extracts a manageable subset in ONE streamed pass so the 5.6GB is never
loaded whole, and saves it to Drive so the big file is needed only once.

Split assignment (baked into a 'split' column, decided once for reproducibility):
  - Training cohorts: N_PER_COHORT patients each, split N_TRAIN train / rest val
    (disjoint people; split-by-person preserved).
  - Holdout cohorts: ALL patients, marked 'holdout' (novel-population robustness test).

Output columns: participant_id, timestamp, glucose, dataset, split
(No insulin/carbs/etc -- DiaData doesn't contain them; glucose-only by nature.)
"""

import numpy as np
import pandas as pd

TRAIN_COHORTS   = ["RBG", "DLCP3", "CITY", "WISDM", "SENCE"]
HOLDOUT_COHORTS = ["HUPA-UCM", "ShanghaiT1D"]
N_PER_COHORT = 35
N_TRAIN = 25          # remaining (10) go to val
SEED = 42


def extract_diadata(csv_path, chunksize=1_000_000):
    # --- Pass 1: cheap scan for patient IDs per cohort (only 2 columns) ---
    ids = pd.read_csv(csv_path, usecols=["PtID", "Database"]).drop_duplicates()

    rng = np.random.RandomState(SEED)
    assign = {}
    for coh in TRAIN_COHORTS:
        pats = sorted(ids[ids["Database"] == coh]["PtID"].unique())
        rng.shuffle(pats)
        chosen = pats[:N_PER_COHORT]
        for i, pid in enumerate(chosen):
            assign[pid] = "train" if i < N_TRAIN else "val"
    for coh in HOLDOUT_COHORTS:
        for pid in ids[ids["Database"] == coh]["PtID"].unique():
            assign[pid] = "holdout"

    print("patients assigned:", pd.Series(assign).value_counts().to_dict())
    keep = set(assign)

    # --- Pass 2: stream the big file, keep only assigned patients ---
    out = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        sub = chunk[chunk["PtID"].isin(keep)].copy()
        if len(sub):
            sub["split"] = sub["PtID"].map(assign)
            out.append(sub)
    result = pd.concat(out, ignore_index=True)

    result = result.rename(columns={"PtID": "participant_id", "ts": "timestamp",
                                    "GlucoseCGM": "glucose", "Database": "dataset"})
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)
    return result[["participant_id", "timestamp", "glucose", "dataset", "split"]]