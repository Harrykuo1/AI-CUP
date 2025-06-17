# plotter.py
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def save_all_plots(fold: int, seed: int, history: dict, log_dir: str = "log"):

    os.makedirs(f"{log_dir}/loss", exist_ok=True)
    os.makedirs(f"{log_dir}/acc", exist_ok=True)
    os.makedirs(f"{log_dir}/auc", exist_ok=True)

    history_tr_loss = history["tr_loss"]
    history_val_loss = history["val_loss"]
    history_acc = history["val_acc_per_task"]
    history_val_auc_gender = history["val_auc_gender"]
    history_val_auc_hand = history["val_auc_hand"]
    history_val_auc_py = history["val_auc_py"]
    history_val_auc_lv = history["val_auc_lv"]
    history_tr_acc_gender = history["tr_acc_gender"]
    history_val_acc_gender = history["val_acc_gender"]
    history_tr_auc_gender = history["tr_auc_gender"]
    history_tr_acc_hand = history["tr_acc_hand"]
    history_val_acc_hand = history["val_acc_hand"]
    history_tr_auc_hand = history["tr_auc_hand"]
    history_tr_acc_py = history["tr_acc_py"]
    history_val_acc_py = history["val_acc_py"]
    history_tr_auc_py = history["tr_auc_py"]
    history_tr_acc_lv = history["tr_acc_lv"]
    history_val_acc_lv = history["val_acc_lv"]
    history_tr_auc_lv = history["tr_auc_lv"]

    epochs = list(range(1, len(history_tr_loss) + 1))

    # --- Plot Loss ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_loss, label="Train Loss")
    plt.plot(epochs, history_val_loss, label="Val   Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/loss/{seed}_{fold}.png")
    plt.close()

    # --- Plot validation ACC for each task ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, [a[0] for a in history_acc], label="gender")
    plt.plot(epochs, [a[1] for a in history_acc], label="hand")
    plt.plot(epochs, [a[2] for a in history_acc], label="play_years")
    plt.plot(epochs, [a[3] for a in history_acc], label="level")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - Validation ACC per Task")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/acc/val_acc_per_task_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot validation AUC for each task ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_val_auc_gender, label="gender")
    plt.plot(epochs, history_val_auc_hand,   label="hand")
    plt.plot(epochs, history_val_auc_py,     label="play_years")
    plt.plot(epochs, history_val_auc_lv,     label="level")
    plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - Validation AUC per Task")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/auc/val_auc_per_task_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot gender ACC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_acc_gender, label="Train ACC gender")
    plt.plot(epochs, history_val_acc_gender, label="Val   ACC gender")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - gender ACC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/acc/gender_acc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot gender AUC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_auc_gender, label="Train AUC gender")
    plt.plot(epochs, history_val_auc_gender, label="Val   AUC gender")
    plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - gender AUC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/auc/gender_auc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot hand ACC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_acc_hand, label="Train ACC hand")
    plt.plot(epochs, history_val_acc_hand, label="Val   ACC hand")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - hand ACC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/acc/hand_acc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot hand AUC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_auc_hand, label="Train AUC hand")
    plt.plot(epochs, history_val_auc_hand, label="Val   AUC hand")
    plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - hand AUC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/auc/hand_auc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot play_years ACC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_acc_py, label="Train ACC play_years")
    plt.plot(epochs, history_val_acc_py, label="Val   ACC play_years")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - play_years ACC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/acc/play_years_acc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot play_years AUC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_auc_py, label="Train AUC play_years")
    plt.plot(epochs, history_val_auc_py, label="Val   AUC play_years")
    plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - play_years AUC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/auc/play_years_auc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot level ACC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_acc_lv, label="Train ACC level")
    plt.plot(epochs, history_val_acc_lv, label="Val   ACC level")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - level ACC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/acc/level_acc_fold{fold}_s{seed}.png")
    plt.close()

    # --- Plot level AUC train vs val ---
    plt.figure()
    ax = plt.gca()
    plt.plot(epochs, history_tr_auc_lv, label="Train AUC level")
    plt.plot(epochs, history_val_auc_lv, label="Val   AUC level")
    plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend()
    plt.title(f"Fold{fold}, Seed{seed} - level AUC")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{log_dir}/auc/level_auc_fold{fold}_s{seed}.png")
    plt.close()