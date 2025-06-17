from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import time
import random
import argparse
import yaml
import gc
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from dataloader import TableTennisDataset, pad_collate
from model_large import (
    TableTennisBig,
    coral_prob,
)

from torch.optim.lr_scheduler import OneCycleLR, LinearLR, CosineAnnealingLR, ExponentialLR
import matplotlib.pyplot as plt
from functools import partial
from plotter import save_all_plots

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def mix_csv(original_csv_path: str, pseudo_csv_path: str, test_sensors_folder: str) -> pd.DataFrame:
    original_df = pd.read_csv(original_csv_path)
    original_df["is_pseudo"]      = 0
    original_df["conf_gender"]     = 1.0
    original_df["conf_hand"]       = 1.0
    original_df["conf_play_years"] = 1.0
    original_df["conf_level"]      = 1.0
    original_df["data_folder"]     = ""
    if os.path.exists(pseudo_csv_path):
        pseudo_df = pd.read_csv(pseudo_csv_path)
        pseudo_df["is_pseudo"]   = 1
        pseudo_df["data_folder"] = test_sensors_folder
        for column in original_df.columns:
            if column not in pseudo_df.columns:
                pseudo_df[column] = np.nan
        pseudo_df = pseudo_df[original_df.columns]
        return pd.concat([original_df, pseudo_df], ignore_index=True)
    return original_df

def compute_loss(
    model: TableTennisBig,
    predictions: dict[str, torch.Tensor],
    labels: torch.Tensor,
    confidences: torch.Tensor,
    pseudo_thresholds: dict[str, float],
    use_pseudo: bool,
    pseudo_weight_high: float,
    pseudo_weight_low: float,
    task_weights: dict[str, float] | None = None,
) -> torch.Tensor:

    device = confidences.device
    total_loss = torch.tensor(0.0, device=device)
    total_weight_sum = 0.0

    def get_pseudo_weight(batch_confidences: torch.Tensor, task_key: str) -> torch.Tensor:
        alpha = ((batch_confidences - pseudo_thresholds[task_key]) / (1 - pseudo_thresholds[task_key])).clamp(0.0, 1.0)
        return pseudo_weight_low + alpha * (pseudo_weight_high - pseudo_weight_low)

    default_task_weights = {"gender": 1.0, "hand": 1.0, "play_years": 1.0, "level": 1.0}
    effective_task_weights = task_weights or default_task_weights

    # gender & hand (二元分類任務)
    for task_index, task_key in enumerate(["gender", "hand"]):
        logits = predictions[task_key]
        batch_labels = labels[:, task_index]
        mask = (batch_labels >= 1) & (batch_labels <= 2)
        if mask.any():
            cross_entropy_loss = F.cross_entropy(logits[mask], batch_labels[mask] - 1, reduction="none")
            if use_pseudo:
                pseudo_weights = get_pseudo_weight(confidences[mask, task_index], task_key).to(device)
                weighted_loss = (cross_entropy_loss * pseudo_weights).mean()
            else:
                weighted_loss = cross_entropy_loss.mean()

            weight = effective_task_weights.get(task_key, 1.0)
            total_loss += weighted_loss * weight
            total_weight_sum += weight

    # play_years (序數迴歸任務)
    labels_play_years = labels[:, 2]
    mask_play_years = labels_play_years >= 0
    if mask_play_years.any():
        if not use_pseudo:
            play_years_loss = model.coral_loss(
                predictions["play_years"][mask_play_years],
                labels_play_years[mask_play_years],
                None
            )
        else:
            raw_logits = predictions["play_years"][mask_play_years]
            label_matrix = (
                torch.arange(raw_logits.size(1), device=raw_logits.device)[None, :]
                < labels_play_years[mask_play_years, None]
            ).float()
            samplewise_bce_loss = F.binary_cross_entropy_with_logits(raw_logits, label_matrix, reduction="none").mean(1)
            pseudo_weights = get_pseudo_weight(confidences[mask_play_years, 2], "play years").to(device)
            play_years_loss = (samplewise_bce_loss * pseudo_weights).mean()

        weight = effective_task_weights.get("play_years", 1.0)
        total_loss += play_years_loss * weight
        total_weight_sum += weight

    # level (序數迴歸任務)
    labels_level = labels[:, 3]
    mask_level = (labels_level >= 2) & (labels_level <= 5)
    if mask_level.any():
        if not use_pseudo:
            level_loss = model.coral_loss(
                predictions["level"][mask_level],
                labels_level[mask_level] - 2,
                None
            )
        else:
            raw_logits = predictions["level"][mask_level]
            label_matrix = (
                torch.arange(raw_logits.size(1), device=raw_logits.device)[None, :]
                < (labels_level[mask_level] - 2)[:, None]
            ).float()
            samplewise_bce_loss = F.binary_cross_entropy_with_logits(raw_logits, label_matrix, reduction="none").mean(1)
            pseudo_weights = get_pseudo_weight(confidences[mask_level, 3], "level").to(device)
            level_loss = (samplewise_bce_loss * pseudo_weights).mean()

        weight = effective_task_weights.get("level", 1.0)
        total_loss += level_loss * weight
        total_weight_sum += weight

    if total_weight_sum > 0:
        return total_loss / total_weight_sum
    else:
        return total_loss


def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)


if __name__ == "__main__":
    script_start_time = time.perf_counter()

    os.makedirs("log", exist_ok=True)
    os.makedirs("log/loss", exist_ok=True)
    os.makedirs("log/acc", exist_ok=True)
    os.makedirs("log/auc", exist_ok=True)
    open("log/logs.txt", "w").close()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--break_epoch", type=int, default=0)
    parser.add_argument("--break_fold", type=int, default=-1)
    parser.add_argument("--pseudo_csv", default="pseudo/pseudo_task.csv")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    model_config = config["model"]
    pseudo_config = config["pseudo"]
    training_config = config["training"]
    break_epoch = args.break_epoch
    break_fold = args.break_fold

    backbone_dropout = model_config["backbone_dropout"]
    attn_dropout = model_config["attn_dropout"]
    head_dropout = model_config["head_dropout"]
    base_weight_decay = config.get("weight_decay", training_config.get("weight_decay", 0.0))
    task_weights = training_config.get("task_weights", None)
    use_mixup = training_config.get("use_mixup", False)
    mixup_alpha = training_config.get("mixup_alpha", 0.0)

    combined_info_df = mix_csv(config["train_info"], args.pseudo_csv, config["test_sensors"])
    
    pseudo_thresholds = {
        "gender":     pseudo_config["thresh"]["bin_hi"],
        "hand":       pseudo_config["thresh"]["bin_hi"],
        "play years": pseudo_config["thresh"]["multi"],
        "level":      pseudo_config["thresh"]["multi"],
    }
    pseudo_weight_high = training_config["pseudo_weight_hi"]
    pseudo_weight_low = training_config["pseudo_weight_lo"]
    pseudo_start_epoch = training_config.get("pseudo_start_epoch", 0)

    real_data_info_df = combined_info_df[combined_info_df.is_pseudo == 0].reset_index(drop=False)
    real_data_labels = real_data_info_df[["gender", "hold racket handed", "play years", "level"]].astype(int).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"#GPUs detected: {num_gpus}")
    use_multi_gpu = num_gpus > 1

    confidence_array = combined_info_df[[
        "conf_gender", "conf_hand", "conf_play_years", "conf_level"
    ]].fillna(0).values

    all_folds_logs = []
    os.makedirs(config["weight_dir"], exist_ok=True)

    for seed in config.get("seed_list", [config["seed"]]):
        seed_all(seed)

        stratified_kfold = MultilabelStratifiedKFold(
            n_splits=config["folds"],
            shuffle=True,
            random_state=seed
        )

        for fold, (train_real_indices_split, val_real_indices_split) in enumerate(stratified_kfold.split(real_data_info_df, real_data_labels)):
            if break_fold != -1 and fold > break_fold:
                print(f"Reached break_fold={break_fold}, forcing next seed")
                break

            real_train_original_indices = real_data_info_df.loc[train_real_indices_split, "index"].tolist()
            real_val_original_indices = real_data_info_df.loc[val_real_indices_split, "index"].tolist()
            pseudo_original_indices = combined_info_df[combined_info_df.is_pseudo == 1].index.tolist()
            
            train_indices = real_train_original_indices + pseudo_original_indices
            val_indices = real_val_original_indices

            dataset = TableTennisDataset(
                config["train_info"], config["train_sensors"],
                ["gender", "hold racket handed", "play years", "level"],
                is_train=True, use_extra_features=config["use_extra_feat"]
            )
            dataset.info = combined_info_df

            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=config["batch_size"], shuffle=True,
                num_workers=config["num_workers"], pin_memory=True,
                collate_fn=pad_collate, worker_init_fn=partial(worker_init_fn, seed=seed)
            )
            
            train_real_subset_indices = [i for i in train_indices if combined_info_df.iloc[i]["is_pseudo"] == 0]
            train_real_only_loader = DataLoader(
                Subset(dataset, train_real_subset_indices),
                batch_size=config["batch_size"], shuffle=False,
                num_workers=config["num_workers"], pin_memory=True,
                collate_fn=pad_collate, worker_init_fn=partial(worker_init_fn, seed=seed)
            )
            
            val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=config["batch_size"], shuffle=False,
                num_workers=config["num_workers"], pin_memory=True,
                collate_fn=pad_collate, worker_init_fn=partial(worker_init_fn, seed=seed)
            )
            
            extra_dim = dataset[0][1][0].shape[0] if config["use_extra_feat"] else 0
            model = TableTennisBig(
                hidden_dim=config["hidden_dim"],
                extra_dim=extra_dim,
                backbone_dropout=backbone_dropout,
                attn_dropout=attn_dropout,
                head_dropout=head_dropout
            ).to(device)
            if use_multi_gpu:
                model = torch.nn.DataParallel(model)
                model.coral_loss = model.module.coral_loss

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=base_weight_decay
            )
            scaler = GradScaler()

            scheduler_config = training_config["scheduler"]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config["lr"] * scheduler_config["max_lr_factor"],
                epochs=config["epochs"],
                steps_per_epoch=len(train_loader),
                pct_start=scheduler_config["pct_start"],
                anneal_strategy=scheduler_config["anneal_strategy"],
                div_factor=scheduler_config["div_factor"],
                final_div_factor=scheduler_config["final_div_factor"],
                three_phase=scheduler_config["three_phase"]
            )

            history_tr_loss, history_val_loss = [], []
            history_val_acc_per_task, history_val_mean_auc = [], []
            history_tr_acc_gender, history_val_acc_gender = [], []
            history_tr_acc_hand,   history_val_acc_hand   = [], []
            history_tr_acc_py,     history_val_acc_py     = [], []
            history_tr_acc_lv,     history_val_acc_lv     = [], []
            history_tr_auc_gender, history_val_auc_gender = [], []
            history_tr_auc_hand,   history_val_auc_hand   = [], []
            history_tr_auc_py,     history_val_auc_py     = [], []
            history_tr_auc_lv,     history_val_auc_lv     = [], []

            best_auc = 0.0
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(1, config["epochs"] + 1):
                if break_epoch != 0 and epoch > break_epoch:
                    print(f"Reached break_epoch={break_epoch}, forcing next fold")
                    break

                for param_group in optimizer.param_groups:
                    param_group["weight_decay"] = base_weight_decay

                epoch_start_time = time.time()
                if epoch == pseudo_start_epoch:
                    log_message = f"=== Pseudo-label enabled at epoch {epoch} ==="
                    print(log_message)
                    open("log/logs.txt","a").write(log_message+"\n")

                use_pseudo = epoch >= pseudo_start_epoch

                model.train()
                total_train_loss, samples_processed = 0.0, 0
                for feats, mask, extras, labels in train_loader:
                    current_batch_size = feats.size(0)
                    feats, mask = feats.to(device), mask.to(device)
                    extras = extras.to(device) if extras is not None else None
                    labels = labels.to(device).long()
                    batch_confidences = torch.tensor(confidence_array[np.array(train_indices)[samples_processed:samples_processed+current_batch_size]], device=device)

                    if use_mixup and mixup_alpha > 0:
                        mixup_lambda = np.random.beta(mixup_alpha, mixup_alpha)
                        mixup_shuffled_indices = torch.randperm(current_batch_size, device=device)
                        feats = feats * mixup_lambda + feats[mixup_shuffled_indices] * (1 - mixup_lambda)
                        shuffled_labels = labels[mixup_shuffled_indices]
                        shuffled_confidences = batch_confidences[mixup_shuffled_indices]
                    else:
                        mixup_lambda = 1.0

                    optimizer.zero_grad()
                    with autocast():
                        predictions = model(feats, mask, extras)
                        loss = compute_loss(
                            model=model, predictions=predictions, labels=labels,
                            confidences=batch_confidences, pseudo_thresholds=pseudo_thresholds,
                            use_pseudo=use_pseudo, pseudo_weight_high=pseudo_weight_high,
                            pseudo_weight_low=pseudo_weight_low, task_weights=task_weights
                        )
                        if mixup_lambda < 1.0:
                            shuffled_loss = compute_loss(
                                model=model, predictions=predictions, labels=shuffled_labels,
                                confidences=shuffled_confidences, pseudo_thresholds=pseudo_thresholds,
                                use_pseudo=use_pseudo, pseudo_weight_high=pseudo_weight_high,
                                pseudo_weight_low=pseudo_weight_low, task_weights=task_weights
                            )
                            loss = mixup_lambda * loss + (1 - mixup_lambda) * shuffled_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if isinstance(scheduler, OneCycleLR):
                        scheduler.step()

                    total_train_loss += loss.item() * current_batch_size
                    samples_processed += current_batch_size
                
                epoch_train_loss = total_train_loss / samples_processed
                history_tr_loss.append(epoch_train_loss)

                model.eval()
                train_predictions_dict = {"gender": [], "hand": [], "play_years": [], "level": []}
                train_true_labels_dict = {"gender": [], "hand": [], "play_years": [], "level": []}
                with torch.no_grad():
                    for feats, mask, extras, labels in train_real_only_loader:
                        feats, mask = feats.to(device), mask.to(device)
                        extras = extras.to(device) if extras is not None else None
                        labels = labels.to(device).long()
                        model_output = model(feats, mask, extras)
                        
                        prob_gender = model_output["gender"].softmax(-1)[:,1].cpu().numpy()
                        prob_hand = model_output["hand"].softmax(-1)[:,1].cpu().numpy()
                        train_predictions_dict["gender"].append(prob_gender)
                        train_true_labels_dict["gender"].append((labels[:,0]-1).cpu().numpy())
                        train_predictions_dict["hand"].append(prob_hand)
                        train_true_labels_dict["hand"].append((labels[:,1]-1).cpu().numpy())
                        
                        pred_probs_play_years = coral_prob(model_output["play_years"]).cpu().numpy()
                        true_labels_play_years = labels[:,2].cpu().numpy().astype(int)
                        train_predictions_dict["play_years"].append(pred_probs_play_years)
                        train_true_labels_dict["play_years"].append(true_labels_play_years)
                        
                        pred_probs_level = coral_prob(model_output["level"]).cpu().numpy()
                        true_labels_level = (labels[:,3].cpu().numpy().astype(int)-2)
                        train_predictions_dict["level"].append(pred_probs_level)
                        train_true_labels_dict["level"].append(true_labels_level)

                train_preds_gender = np.concatenate(train_predictions_dict["gender"])
                train_trues_gender = np.concatenate(train_true_labels_dict["gender"])
                history_tr_acc_gender.append(accuracy_score(train_trues_gender, (train_preds_gender >= 0.5).astype(int)))
                history_tr_auc_gender.append(roc_auc_score(train_trues_gender, train_preds_gender))

                train_preds_hand = np.concatenate(train_predictions_dict["hand"])
                train_trues_hand = np.concatenate(train_true_labels_dict["hand"])
                history_tr_acc_hand.append(accuracy_score(train_trues_hand, (train_preds_hand >= 0.5).astype(int)))
                history_tr_auc_hand.append(roc_auc_score(train_trues_hand, train_preds_hand))

                train_preds_play_years = np.concatenate(train_predictions_dict["play_years"], axis=0)
                train_trues_play_years = np.concatenate(train_true_labels_dict["play_years"], axis=0)
                history_tr_acc_py.append(accuracy_score(train_trues_play_years, train_preds_play_years.argmax(1)))
                history_tr_auc_py.append(
                    roc_auc_score(pd.get_dummies(train_trues_play_years).values, train_preds_play_years,
                                  multi_class="ovr", average="micro")
                )

                train_preds_level = np.concatenate(train_predictions_dict["level"], axis=0)
                train_trues_level = np.concatenate(train_true_labels_dict["level"], axis=0)
                history_tr_acc_lv.append(accuracy_score(train_trues_level, train_preds_level.argmax(1)))
                history_tr_auc_lv.append(
                    roc_auc_score(pd.get_dummies(train_trues_level).values, train_preds_level,
                                  multi_class="ovr", average="micro")
                )

                val_predictions_dict = {k:[] for k in ["gender","hand","play_years","level"]}
                val_true_labels_dict = {k:[] for k in ["gender","hand","play_years","level"]}
                total_val_loss, val_samples_processed = 0.0, 0
                with torch.no_grad():
                    for feats, mask, extras, labels in val_loader:
                        current_batch_size = feats.size(0)
                        feats, mask = feats.to(device), mask.to(device)
                        extras = extras.to(device) if extras is not None else None
                        labels = labels.to(device).long()
                        batch_confidences = torch.tensor(confidence_array[np.array(val_indices)[val_samples_processed:val_samples_processed+current_batch_size]], device=device)

                        model_output = model(feats, mask, extras)
                        batch_val_loss = compute_loss(
                            model=model, predictions=model_output, labels=labels,
                            confidences=batch_confidences, pseudo_thresholds=pseudo_thresholds,
                            use_pseudo=use_pseudo, pseudo_weight_high=pseudo_weight_high,
                            pseudo_weight_low=pseudo_weight_low, task_weights=task_weights
                        )
                        total_val_loss += batch_val_loss.item() * current_batch_size
                        val_samples_processed += current_batch_size

                        labels_numpy = labels.cpu().numpy()
                        val_predictions_dict["gender"].append(model_output["gender"].softmax(-1).cpu().numpy()[:,1])
                        val_true_labels_dict["gender"].append(labels_numpy[:,0]-1)
                        val_predictions_dict["hand"].append(model_output["hand"].softmax(-1).cpu().numpy()[:,1])
                        val_true_labels_dict["hand"].append(labels_numpy[:,1]-1)
                        pred_probs_play_years = coral_prob(model_output["play_years"]).cpu().numpy()
                        val_predictions_dict["play_years"].append(pred_probs_play_years)
                        val_true_labels_dict["play_years"].append(labels_numpy[:,2])
                        pred_probs_level = coral_prob(model_output["level"]).cpu().numpy()
                        val_predictions_dict["level"].append(pred_probs_level)
                        val_true_labels_dict["level"].append(labels_numpy[:,3]-2)

                epoch_val_loss = total_val_loss / val_samples_processed
                history_val_loss.append(epoch_val_loss)

                val_aucs = {}
                for task_key in ["gender","hand"]:
                    val_aucs[task_key] = roc_auc_score(np.concatenate(val_true_labels_dict[task_key]), np.concatenate(val_predictions_dict[task_key]))
                val_aucs["play_years"] = roc_auc_score(
                    pd.get_dummies(np.concatenate(val_true_labels_dict["play_years"])).values,
                    np.concatenate(val_predictions_dict["play_years"]), multi_class="ovr", average="micro"
                )
                val_aucs["level"] = roc_auc_score(
                    pd.get_dummies(np.concatenate(val_true_labels_dict["level"])).values,
                    np.concatenate(val_predictions_dict["level"]), multi_class="ovr", average="micro"
                )
                history_val_auc_gender.append(val_aucs["gender"])
                history_val_auc_hand.append(val_aucs["hand"])
                history_val_auc_py.append(val_aucs["play_years"])
                history_val_auc_lv.append(val_aucs["level"])

                acc_gender = accuracy_score(
                    np.concatenate(val_true_labels_dict["gender"]), (np.concatenate(val_predictions_dict["gender"])>=0.5).astype(int)
                )
                acc_hand = accuracy_score(
                    np.concatenate(val_true_labels_dict["hand"]), (np.concatenate(val_predictions_dict["hand"])>=0.5).astype(int)
                )
                acc_py = accuracy_score(
                    np.concatenate(val_true_labels_dict["play_years"]), np.argmax(np.concatenate(val_predictions_dict["play_years"]), axis=1)
                )
                acc_lv = accuracy_score(
                    np.concatenate(val_true_labels_dict["level"]), np.argmax(np.concatenate(val_predictions_dict["level"]), axis=1)
                )
                history_val_acc_gender.append(acc_gender)
                history_val_acc_hand.append(acc_hand)
                history_val_acc_py.append(acc_py)
                history_val_acc_lv.append(acc_lv)
                
                history_val_acc_per_task.append([acc_gender, acc_hand, acc_py, acc_lv,
                                                 np.mean([acc_gender, acc_hand, acc_py, acc_lv])])
                history_val_mean_auc.append(np.mean(list(val_aucs.values())))

                log_line = (
                    f"Seed:{seed} | Fold:{fold} | Ep:{epoch} | "
                    f"TrL:{epoch_train_loss:.4f} | ValL:{epoch_val_loss:.4f} | "
                    f"AUCs:{val_aucs['gender']:.4f},{val_aucs['hand']:.4f},"
                    f"{val_aucs['play_years']:.4f},{val_aucs['level']:.4f} | "
                    f"MeanAUC:{np.mean(list(val_aucs.values())):.4f} | "
                    f"LR:{optimizer.param_groups[0]['lr']:.6f}"
                )
                print(log_line)
                with open("log/logs.txt","a") as f:
                    f.write(log_line + "\n")

                auc_improvement_threshold = 0.000099
                current_mean_auc = np.mean(list(val_aucs.values()))
                if (current_mean_auc > best_auc) or \
                   (abs(current_mean_auc - best_auc) <= auc_improvement_threshold and epoch_val_loss < best_loss):
                    best_auc = current_mean_auc
                    best_loss = epoch_val_loss
                    patience_counter = 0
                    torch.save(getattr(model, 'module', model).state_dict(),
                               os.path.join(config["weight_dir"], f"fold{fold}_s{seed}.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= config["patience"]:
                        print(f"Early stop @ Ep{epoch}")
                        break

            history_data = {
                "tr_loss": history_tr_loss,
                "val_loss": history_val_loss,
                "val_acc_per_task": history_val_acc_per_task,
                "val_auc_gender": history_val_auc_gender,
                "val_auc_hand": history_val_auc_hand,
                "val_auc_py": history_val_auc_py,
                "val_auc_lv": history_val_auc_lv,
                "tr_acc_gender": history_tr_acc_gender,
                "val_acc_gender": history_val_acc_gender,
                "tr_auc_gender": history_tr_auc_gender,
                "tr_acc_hand": history_tr_acc_hand,
                "val_acc_hand": history_val_acc_hand,
                "tr_auc_hand": history_tr_auc_hand,
                "tr_acc_py": history_tr_acc_py,
                "val_acc_py": history_val_acc_py,
                "tr_auc_py": history_tr_auc_py,
                "tr_acc_lv": history_tr_acc_lv,
                "val_acc_lv": history_val_acc_lv,
                "tr_auc_lv": history_tr_auc_lv,
            }

            save_all_plots(fold=fold, seed=seed, history=history_data, log_dir="log")

            model.cpu()
            del model, optimizer, scheduler, scaler, train_loader, train_real_only_loader, val_loader, dataset
            torch.cuda.empty_cache()
            gc.collect()

            all_folds_logs.append({
                "fold": fold, "seed": seed,
                "auc_gender":     val_aucs["gender"],
                "auc_hand":       val_aucs["hand"],
                "auc_play_years": val_aucs["play_years"],
                "auc_level":      val_aucs["level"],
            })

    os.makedirs("result", exist_ok=True)
    pd.DataFrame(all_folds_logs).to_csv("result/val_auc.csv", index=False)
    print("Saved result/val_auc.csv")

    script_end_time = time.perf_counter()
    final_log_message = f"總訓練時間：{(script_end_time - script_start_time)/60:.2f} 分鐘"
    with open("log/logs.txt","a") as f:
        f.write(final_log_message + "\n")
    print(final_log_message)