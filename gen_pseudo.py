# gen_pseudo.py

import os
import glob
import argparse
import yaml
import pandas as pd
import numpy as np

BIN_TASKS = ["gender", "hold racket handed"]
MULTI_TASK = {
    "play years": ["play years_0", "play years_1", "play years_2"],
    "level":      ["level_2",     "level_3",     "level_4",     "level_5"],
}

def decide_binary(probs, hi, lo):
    if all(p >= hi for p in probs): 
        return 1
    if all(p <= lo for p in probs): 
        return 2
    return 0

def decide_multi(plist, thr):
    chosen = [(int(np.argmax(p)), float(p[np.argmax(p)])) for p in plist]
    idx0 = chosen[0][0]
    if all(idx == idx0 and prob >= thr for idx, prob in chosen):
        return idx0
    return -1

def load_submissions(pdir, prefix):
    pattern = os.path.join(pdir, f"{prefix}*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No submissions found matching {pattern}")
    subs = [pd.read_csv(f) for f in files]
    u0 = subs[0]["unique_id"].tolist()
    for df in subs[1:]:
        if df["unique_id"].tolist() != u0:
            raise ValueError("All submissions must have identical unique_id ordering")
    print(f"Loaded {len(subs)} submissions")
    return subs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pcfg = cfg["pseudo"]

    # load submissions
    subs      = load_submissions(pcfg["pseudo_dir"], pcfg["pseudo_prefix"])
    uids      = subs[0]["unique_id"].tolist()
    bin_hi    = pcfg["thresh"]["bin_hi"]
    bin_lo    = pcfg["thresh"]["bin_lo"]
    multi_thr = pcfg["thresh"]["multi"]
    ratio     = float(pcfg["ratio"])

    test_info  = pd.read_csv(cfg["test_info"])
    train_info = pd.read_csv(cfg["train_info"]).set_index("unique_id")
    real_n     = len(train_info)
    max_keep   = int(real_n * ratio)

    records = []
    decs    = []
    confs   = []
    pre_counts = {t: 0 for t in BIN_TASKS + list(MULTI_TASK.keys())}

    for i, uid in enumerate(uids):
        uid = int(uid)
        # gather probabilities from each submission
        bin_p = {t: [df.at[i, t] for df in subs] for t in BIN_TASKS}
        multi_p = {
            t: [np.array([df.at[i, c] for c in cols]) for df in subs]
            for t, cols in MULTI_TASK.items()
        }

        # decide labels
        dec = {}
        for t in BIN_TASKS:
            d = decide_binary(bin_p[t], bin_hi, bin_lo)
            dec[t] = d
            if d in (1, 2):
                pre_counts[t] += 1

        for t in MULTI_TASK:
            idx = decide_multi(multi_p[t], multi_thr)
            if t == "level":
                d = idx + 2 if idx >= 0 else -1
            else:
                d = idx
            dec[t] = d
            if d != -1:
                pre_counts[t] += 1

        # skip if no task passed
        if all(dec[t] in (0, -1) for t in dec):
            continue

        # compute per-task confidence
        conf = {}
        for t in BIN_TASKS:
            d = dec[t]
            if d == 1:
                # positive class mean probability
                conf[t] = float(np.mean(bin_p[t]))
            elif d == 2:
                # negative class mean probability = mean(1 - p)
                conf[t] = float(np.mean([1 - p for p in bin_p[t]]))
            else:
                conf[t] = 0.0

        for t in MULTI_TASK:
            d = dec[t]
            if d >= 0:
                idx0 = d - 2 if t == "level" else d
                conf[t] = float(np.mean([p[idx0] for p in multi_p[t]]))
            else:
                conf[t] = 0.0

        # meta info
        row = test_info[test_info["unique_id"] == uid]
        mode = int(row["mode"].iloc[0]) if not row.empty else -1
        cp   = row["cut_point"].iloc[0] if "cut_point" in row else ""
        pid  = int(train_info.at[uid, "player_id"]) if uid in train_info.index else -1

        # record including confidence columns
        records.append({
            "unique_id":           uid,
            "player_id":           pid,
            "mode":                mode,
            "gender":              dec["gender"],
            "conf_gender":         conf["gender"],
            "hold racket handed":  dec["hold racket handed"],
            "conf_hand":           conf["hold racket handed"],
            "play years":          dec["play years"],
            "conf_play_years":     conf["play years"],
            "level":               dec["level"],
            "conf_level":          conf["level"],
            "cut_point":           cp,
            "data_folder":         cfg["test_sensors"]
        })
        decs.append(dec)
        confs.append(conf)

    df = pd.DataFrame(records)

    # collect per-task indices and apply cap
    task_indices = {}
    for t in pre_counts:
        if t in BIN_TASKS:
            mask = [dec[t] in (1, 2) for dec in decs]
        else:
            mask = [dec[t] != -1 for dec in decs]
        idxs = np.where(mask)[0]
        sorted_idxs = sorted(idxs, key=lambda i: confs[i][t], reverse=True)
        task_indices[t] = set(sorted_idxs[:max_keep])

    keep_union = set().union(*task_indices.values())
    post_counts = {t: len(task_indices[t]) for t in task_indices}

    # logging
    print("=== Pseudo Label Statistics per Task ===")
    for t in pre_counts:
        print(f"  {t:<22}: passed pre-cap {pre_counts[t]:>4}, post-cap {post_counts[t]:>4}")
    print(f"Total candidates: {len(records)}")
    print(f"Total selected:   {len(keep_union)}")

    # save unioned pseudo labels with confidence
    out_df = df.iloc[sorted(keep_union)].reset_index(drop=True)
    out_csv = os.path.join(pcfg["pseudo_dir"], "pseudo_task.csv")
    os.makedirs(pcfg["pseudo_dir"], exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

if __name__ == "__main__":
    main()
