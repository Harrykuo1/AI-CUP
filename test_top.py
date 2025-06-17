import os
import argparse
import yaml
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from dataloader import TableTennisDataset, pad_collate_eval
from train import seed_all
from model_large import TableTennisBig, coral_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="設定檔路徑")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--top_frac", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    seed_all(int(config.get("seed", 42)))

    batch_size = int(args.batch_size if args.batch_size else config["batch_size"])
    hidden_dim = int(config["hidden_dim"])
    num_workers = int(config["num_workers"])
    use_extra_features = bool(config["use_extra_feat"])
    weight_dir = config["weight_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validation_auc_df = pd.read_csv("result/val_auc.csv")

    path_pattern = re.compile(r"fold(\d+)_s(\d+)\.pth$")
    model_paths, model_keys = [], []
    for weight_path in sorted(glob.glob(os.path.join(weight_dir, "fold*_s*.pth"))):
        path_match = path_pattern.search(os.path.basename(weight_path))
        if not path_match:
            continue
        
        fold, seed = int(path_match.group(1)), int(path_match.group(2))
        if args.folds is None or fold in args.folds:
            model_paths.append(weight_path)
            model_keys.append((fold, seed))
            
    total_models = len(model_paths)
    if total_models == 0:
        raise RuntimeError(f"在 '{weight_dir}' 中找不到任何模型權重檔案")

    print(f"使用 Batch Size={batch_size}, 共找到 {total_models} 個模型進行推論")
    torch.cuda.empty_cache()

    tasks = ["gender", "hand", "play_years", "level"]
    selected_model_indices_per_task = {}
    for task_key in tasks:
        auc_column = f"auc_{task_key}"
        auc_scores = [
            float(validation_auc_df[(validation_auc_df.fold == f) & (validation_auc_df.seed == s)].iloc[0][auc_column])
            if not validation_auc_df[(validation_auc_df.fold == f) & (validation_auc_df.seed == s)].empty else 0.0
            for f, s in model_keys
        ]
        
        sorted_model_indices = np.argsort(auc_scores)[::-1]
        num_selected_models = max(1, int(np.ceil(args.top_frac * total_models)))
        selected_model_indices_per_task[task_key] = set(sorted_model_indices[:num_selected_models])
        print(f"任務 '{task_key}': 選擇了 {num_selected_models}/{total_models} 個模型")

    test_dataset = TableTennisDataset(
        config["test_info"], config["test_sensors"],
        label_cols=None, is_train=False, use_extra_features=use_extra_features
    )
    extra_dim = 0
    if use_extra_features:
        _, extras_sample, _ = test_dataset[0]
        if extras_sample:
            extra_dim = extras_sample[0].shape[0]

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=pad_collate_eval
    )

    models = []
    for weight_path in model_paths:
        model_instance = TableTennisBig(hidden_dim, extra_dim).to(device).eval().half()
        model_instance.load_state_dict(torch.load(weight_path, map_location=device))
        models.append(model_instance)

    TTA_SHIFTS = [-3, 0, 3]

    feats_sample, mask_sample, extras_sample, _ = next(iter(test_loader))
    feats_sample = feats_sample.to(device).half()
    mask_sample = mask_sample.to(device)
    extras_sample = extras_sample.to(device).half() if extras_sample is not None else None
    
    output_sample = models[0](feats_sample, mask_sample, extras_sample)
    output_dims = {
        task_key: (coral_prob(value_tensor).shape[-1] if task_key in ["play_years", "level"]
                   else value_tensor.softmax(-1).shape[-1])
        for task_key, value_tensor in output_sample.items()
    }

    records = []
    with torch.no_grad():
        for batch_feats, batch_mask, batch_extras, _ in test_loader:
            current_batch_size = batch_feats.size(0)
            batch_feats = batch_feats.to(device).half()
            batch_mask = batch_mask.to(device)
            batch_extras = batch_extras.to(device).half() if batch_extras is not None else None

            batch_predictions_sum = {task_key: np.zeros((current_batch_size, output_dims[task_key]), dtype=np.float64) for task_key in tasks}

            for shift_value in TTA_SHIFTS:
                shifted_feats = torch.roll(batch_feats, shifts=shift_value, dims=-1)
                for model_index, model in enumerate(models):
                    for task_key in tasks:
                        if model_index not in selected_model_indices_per_task[task_key]:
                            continue
                        
                        model_output = model(shifted_feats, batch_mask, batch_extras)
                        if task_key in ["play_years", "level"]:
                            probabilities = coral_prob(model_output[task_key]).cpu().numpy()
                        else:
                            probabilities = model_output[task_key].softmax(-1).cpu().numpy()
                        batch_predictions_sum[task_key] += probabilities

            for task_key in tasks:
                denominator = len(selected_model_indices_per_task[task_key]) * len(TTA_SHIFTS)
                batch_predictions_sum[task_key] /= denominator
                if task_key in ["play_years", "level"]:
                    row_sums = batch_predictions_sum[task_key].sum(axis=1, keepdims=True)
                    batch_predictions_sum[task_key] /= row_sums

            offset = len(records)
            for i in range(current_batch_size):
                unique_id = test_dataset.info["unique_id"].iloc[offset + i]
                record = {
                    "unique_id": unique_id,
                    "gender": float(batch_predictions_sum["gender"][i, 0].astype(np.float32)),
                    "hold racket handed": float(batch_predictions_sum["hand"][i, 0].astype(np.float32)),
                }
                for j in range(output_dims["play_years"]):
                    record[f"play years_{j}"] = float(batch_predictions_sum["play_years"][i, j].astype(np.float32))
                for j in range(output_dims["level"]):
                    record[f"level_{j+2}"] = float(batch_predictions_sum["level"][i, j].astype(np.float32))
                records.append(record)

    pd.DataFrame(records).to_csv("submission_top_models.csv", index=False, float_format="%.8f")
    print("Save submission_top_models.csv")

if __name__ == "__main__":
    main()