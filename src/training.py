import os
import math
from time import time
from os.path import join
from collections import defaultdict
from typing import Optional, Iterator

import torch
import optuna
import kagglehub 
import numpy as np
import pandas as pd
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
import torch.multiprocessing as mp
from pandas import DataFrame as DF
from torch.utils.data import Subset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as DL
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

from config import *
from model import mk_model
from utils import seed_everything
from preprocessing import get_meta_data


class CMIDataset(TensorDataset):
    def __init__(self, device: torch.device):
        x = np.load(join("preprocessed_dataset", "X.npy")).swapaxes(1, 2)
        y = np.load(join("preprocessed_dataset", "Y.npy"))
        auxiliary_orientation_y = np.load(join("preprocessed_dataset", "orientation_Y.npy"))
        auxiliary_demographics_y = np.load(join("preprocessed_dataset", "demographics_Y.npy"))
        super().__init__(
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(auxiliary_orientation_y).to(device),
            torch.from_numpy(auxiliary_demographics_y).to(device),
        )

    def __getitem__(self, index):
        return *super().__getitem__(index), index

class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_lr: float,
        min_lr: float,
        cycle_length: int,
        cycle_mult: float = 1.0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for linear warmup.
            max_lr: Initial maximum learning rate.
            min_lr: Minimum learning rate after decay.
            cycle_length: Initial number of steps per cosine cycle.
            cycle_mult: Multiplicative factor for increasing cycle lengths.
            gamma: Multiplicative decay factor for max_lr after each cycle.
            last_epoch: The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.gamma = gamma

        self.current_cycle = 0
        self.cycle_step = 0
        self.lr = max_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [self.min_lr + scale * (self.max_lr - self.min_lr) for _ in self.base_lrs]

        # Adjust for post-warmup step index
        t = self.cycle_step
        T = self.cycle_length

        cosine_decay = 0.5 * (1 + math.cos(math.pi * t / T))
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return [lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None) -> None:
        if self.last_epoch >= self.warmup_steps:
            self.cycle_step += 1
            if self.cycle_step >= self.cycle_length:
                self.current_cycle += 1
                self.cycle_step = 0
                self.cycle_length = max(int(self.cycle_length * self.cycle_mult), 1)
                self.max_lr *= self.gamma
        super().step(epoch)

def mk_scheduler(optimizer:Optimizer, steps_per_epoch:int, lr_scheduler_kw:dict) -> _LRScheduler:
    return CosineAnnealingWarmupRestarts(
        optimizer,
        warmup_steps=lr_scheduler_kw["warmup_epochs"] * steps_per_epoch,
        cycle_mult=lr_scheduler_kw["cycle_mult"],
        max_lr=lr_scheduler_kw["max_lr"],
        min_lr=lr_scheduler_kw["max_lr"] / lr_scheduler_kw["max_to_min_div_factor"],
        cycle_length=lr_scheduler_kw["init_cycle_epochs"] * steps_per_epoch,
        gamma=lr_scheduler_kw["lr_cycle_factor"],
    ) 

def mixup_data(
        x:Tensor,
        y:Tensor,
        aux_y:Tensor,
        alpha=0.2
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    mixed_aux_y = lam * aux_y + (1 - lam) * aux_y[index, :]

    return mixed_x, mixed_y, mixed_aux_y

def train_model_on_single_epoch(
        meta_data:dict,
        model:nn.Module,
        train_loader:DL,
        criterion:callable,
        optimizer:torch.optim.Optimizer,
        scheduler:_LRScheduler,
        training_kw:dict,
        device:torch.device,
    ) -> dict:
    "Train model on a single epoch"
    train_metrics = {}
    model.train()
    train_metrics["train_loss"] = 0.0
    total = 0
    for batch_x, batch_y, batch_orientation_y, demographics_y, idx in train_loader:
        batch_orientation_y = batch_orientation_y.clone()
        batch_x = batch_x.to(device).clone()
        add_noise = torch.randn_like(batch_x, device=device) * 0.04
        scale_noise = torch.rand_like(batch_x, device=device) * (1.1 - 0.9) + 0.9
        batch_x = (add_noise + batch_x) * scale_noise
        batch_x[:TRAIN_BATCH_SIZE // 2, meta_data["tof_idx"] + meta_data["thm_idx"]] = 0.0
        batch_y = batch_y.to(device)
        batch_x = batch_x.float()
        
        batch_x, batch_y, batch_orientation_y = mixup_data(batch_x, batch_y, batch_orientation_y)

        optimizer.zero_grad()
        outputs, orient_output, demos_output = model(batch_x)
        bce = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, batch_y)  \
               + bce(demos_output, demographics_y) * training_kw["demos_loss_weight"] \
               + criterion(orient_output, batch_orientation_y) * training_kw["orient_loss_weight"] \
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_metrics["train_loss"] += loss.item() * batch_x.size(0)
        total += batch_x.size(0)

    train_metrics["train_loss"] /= total

    return train_metrics

def evaluate_model(
        meta_data:dict,
        model:nn.Module,
        validation_loader:DL,
        criterion:callable,
        device:torch.device
    ) -> dict:
    model.eval()
    eval_metrics = {}
    eval_metrics["val_loss"] = 0.0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_x, batch_y, oirent_y, demos_y, idx in validation_loader:
            batch_x = batch_x.to(device).clone()
            batch_y = batch_y.to(device)
            batch_x[:VALIDATION_BATCH_SIZE // 2, meta_data["tof_idx"] + meta_data["thm_idx"]] = 0.0

            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            eval_metrics["val_loss"] += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # Get true class indices from one-hot
            trues = torch.argmax(batch_y, dim=1).cpu().numpy()

            all_true.append(trues)
            all_pred.append(preds)

    eval_metrics["val_loss"] /= total
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Compute competition metrics
    # Binary classification: BFRB (1) vs non-BFRB (0)
    binary_true = np.isin(all_true, BFRB_INDICES).astype(int)
    binary_pred = np.isin(all_pred, BFRB_INDICES).astype(int)
    eval_metrics["binary_f1"] = f1_score(binary_true, binary_pred)

    # Collapse non-BFRB gestures into a single class
    collapsed_true = np.where(
        np.isin(all_true, BFRB_INDICES),
        all_true,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )
    collapsed_pred = np.where(
        np.isin(all_pred, BFRB_INDICES),
        all_pred,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )

    # Macro F1 on collapsed classes
    eval_metrics["macro_f1"] = f1_score(collapsed_true, collapsed_pred, average='macro')
    eval_metrics["final_metric"] = (eval_metrics["binary_f1"] + eval_metrics["macro_f1"]) / 2

    return eval_metrics

def get_perf_and_seq_id(model:nn.Module, data_loader:DL, device:torch.device) -> DF:
    metrics:dict[list[ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_aux_y, batch_orient_y, idx in data_loader:
            batch_x = batch_x.to(device).clone()
            batch_y = batch_y.to(device)
            # batch_x[:VALIDATION_BATCH_SIZE // 2, tof_idx + thm_idx] = 0.0

            outputs, orient_outputs, demos_output = model(batch_x)
            losses = nn.functional.cross_entropy(
                outputs,
                batch_y,
                label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            )
            orient_losses = nn.functional.cross_entropy(
                orient_outputs,
                batch_aux_y,
                label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            )
            demos_losses = nn.functional.binary_cross_entropy_with_logits(
                demos_output,
                batch_orient_y,
                # label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            ).cpu().numpy()
            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # Get true class indices from one-hot
            trues = torch.argmax(batch_y, dim=1).cpu().numpy()
            accuracies = (preds == trues) #.cpu().numpy()

            metrics["losses"].append(losses.cpu().numpy())
            metrics["orient_losses"].append(orient_losses.cpu().numpy())
            for denos_target_idx, demos_target in enumerate(DEMOS_TARGETS):
                metrics[f"{demos_target}_losses"].append(demos_losses[:, denos_target_idx])
            metrics["preds"].append(preds)
            metrics["trues"].append(trues)
            metrics["accuracies"].append(accuracies)
            metrics["sequence_id"].append(seq_meta_data["sequence_id"].iloc[idx].values)

    metrics = {k: np.concat(v) for k, v in metrics.items()}

    return DF.from_records(metrics)

def get_per_sequence_meta_data(model:nn.Module, train_dataset:Dataset, val_dataset:Dataset, device:torch.device) -> DF:
    train_DL = DL(train_dataset, VALIDATION_BATCH_SIZE, shuffle=False)
    val_DL = DL(val_dataset, VALIDATION_BATCH_SIZE, shuffle=False)
    return pd.concat((
        get_perf_and_seq_id(model, train_DL, device).assign(is_train=True),
        get_perf_and_seq_id(model, val_DL, device).assign(is_train=False),
    ))

def train_model_on_all_epochs(
        model:nn.Module,
        train_dataset:Dataset,
        validation_dataset:Dataset,
        criterion:callable,
        optimizer:torch.optim.Optimizer,
        scheduler:_LRScheduler,
        fold:int,
        training_kw:dict,
        device:torch.device,
    ) -> tuple[DF, DF]:
    """
    Returns:
        tuple[DF, DF]: epoch wise metrics, sample(sequence) wise meta data metrics
    """
    meta_data = get_meta_data()
    train_loader = DL(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DL(validation_dataset, VALIDATION_BATCH_SIZE, shuffle=False, drop_last=False)

    metrics:list[dict] = []
    # Early stopping
    best_metric = -np.inf
    epochs_no_improve = 0

    epoch = 0
    for _ in range(TRAINING_EPOCHS):
        epoch += 1
        train_metrics = train_model_on_single_epoch(meta_data, model, train_loader, criterion, optimizer, scheduler, training_kw, device)
        validation_metrics = evaluate_model(meta_data, model, validation_loader, criterion, device)
        metrics.append({"fold": fold, "epoch": epoch} | train_metrics | validation_metrics)
        if validation_metrics["final_metric"] > best_metric:
            best_metric = validation_metrics["final_metric"]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                model.load_state_dict(best_model_state)
                break
    print("fold", fold, "stopped at", epoch, "score:", validation_metrics['final_metric'])

    os.makedirs("models", exist_ok=True)
    torch.save(best_model_state, f"models/model_fold_{fold}.pth")
    epoch_wise_metrics = DF.from_records(metrics).set_index(["fold", "epoch"])
    seq_meta_data = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    sample_wise_meta_data = (
        get_per_sequence_meta_data(model, train_dataset, validation_dataset, device)
        .merge(seq_meta_data, how="left", on="sequence_id")
    )

    return epoch_wise_metrics, sample_wise_meta_data

def train_on_single_fold(
        fold_idx:int,
        full_dataset:TensorDataset,
        train_idx:ndarray,
        validation_idx:ndarray,
        lr_scheduler_kw: dict,
        optimizer_kw: dict,
        training_kw: dict,
        seed:int,
        gpu_id:int,
    ) -> tuple[DF, DF]:
    seed_everything(seed)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    train_dataset = Subset(full_dataset, train_idx)
    validation_dataset = Subset(full_dataset, validation_idx)
    all_train_x = train_dataset.dataset.tensors[0][train_dataset.indices]
    device = torch.device(f"cuda:{gpu_id}")
    model = mk_model(all_train_x, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        WARMUP_LR_INIT,
        weight_decay=optimizer_kw["weight_decay"],
        betas=(optimizer_kw["beta_0"], optimizer_kw["beta_1"]),
    )
    steps_per_epoch = len(DL(train_dataset, TRAIN_BATCH_SIZE)) # ugly, i know
    scheduler = mk_scheduler(optimizer, steps_per_epoch, lr_scheduler_kw)
    epoch_metrics, seq_metrics = train_model_on_all_epochs(
        model,
        train_dataset,
        validation_dataset,
        criterion,
        optimizer,
        scheduler,
        fold_idx,
        training_kw,
        device=device,
    )
    os.makedirs("metrics/", exist_ok=True)
    epoch_metrics.to_parquet(f"metrics/epoch_metrics_fold_{fold_idx}.parquet")
    seq_metrics.to_parquet(f"metrics/seq_metrics_fold_{fold_idx}.parquet")

def sgkf_from_tensor_dataset(
    # dataset: TensorDataset,
    n_splits: int = 5,
    shuffle: bool = True,
) -> Iterator[tuple[int, int, int]]:
    """Returns iterator of tuple of train and val indices and seed."""
    # Load sequence meta data to get classes and groups parameters
    seq_meta = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    # X, *_ = dataset.tensors
    x = np.load("preprocessed_dataset/X.npy")
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
    )

    fold_indices = list(sgkf.split(x, seq_meta["gesture"], seq_meta["subject"]))
    folds_idx_oredered_by_score:list[int] = FOLDS_VAL_SCORE_ORDER.get(N_FOLDS, range(N_FOLDS))

    for fold_idx in folds_idx_oredered_by_score:
        yield *fold_indices[fold_idx], SEED + fold_idx

def load_metrics(name_format:str) -> DF:
    all_metrics = DF()
    for fold_idx in range(N_FOLDS):
        fold_metrics = pd.read_parquet(join("metrics", name_format.format(fold_idx)))
        all_metrics = pd.concat((all_metrics, fold_metrics))
    return all_metrics

def train_on_all_folds(
        lr_scheduler_kw: dict,
        optimizer_kw: dict,
        training_kw: dict,
        trial: Optional[optuna.trial.Trial]=None,
    ) -> None:
    start_time = time()
    seed_everything(seed=SEED)
    ctx = mp.get_context("spawn")
    gpus = list(range(torch.cuda.device_count()))
    full_datasets = {gpu_idx: CMIDataset(torch.device(f"cuda:{gpu_idx}")) for gpu_idx in gpus}

    folds_it = list(sgkf_from_tensor_dataset(N_FOLDS))
    processes: list[mp.Process] = []

    # keep track of which GPU is free
    active: dict[int, mp.Process] = {}

    for fold_idx, (train_idx, validation_idx, seed) in enumerate(folds_it):
        # wait until a GPU is free
        while True:
            free_gpus = [gpu for gpu in gpus if gpu not in active]
            if free_gpus:
                gpu_idx = free_gpus[0]
                break
            # block until one process finishes
            for gpu_idx, proc in list(active.items()):
                proc.join(timeout=0.1)
                if not proc.is_alive():
                    proc.close()
                    del active[gpu_idx]

        # now gpu_idx is free
        p = ctx.Process(
            target=train_on_single_fold,
            args=(
                fold_idx,
                full_datasets[gpu_idx],
                train_idx,
                validation_idx,
                lr_scheduler_kw,
                optimizer_kw,
                training_kw,
                seed,
                gpu_idx,
            )
        )
        # print(f"Starting process for fold {fold_idx} on GPU {gpu_idx}")
        p.start()
        active[gpu_idx] = p
        processes.append(p)

    # wait for all remaining processes
    for gpu_idx, proc in active.items():
        proc.join()
        proc.close()

    end_time = time()
    
    epoch_metrics = load_metrics("epoch_metrics_fold_{}.parquet")
    seq_metrics = load_metrics("seq_metrics_fold_{}.parquet")
    mean_val_score = (
        epoch_metrics
        .groupby(level=0)
        .agg({"final_metric": "max"})
        .mean()
    )
    print(f"done in {end_time - start_time:.2f}s, mean score:", mean_val_score)
    print("all epochs metrics:")
    print(epoch_metrics)
    print("")
    return mean_val_score, epoch_metrics, seq_metrics

if __name__ == "__main__":
    mean_val_score, epoch_metrics, seq_metrics = train_on_all_folds(
        lr_scheduler_kw={
            'warmup_epochs': 14,
            'cycle_mult': 0.9,
            'max_lr': 0.00652127195137508,
            'init_cycle_epochs': 4,
            'lr_cycle_factor': 0.45,
            'max_to_min_div_factor': 250,
        },
        optimizer_kw={
            'weight_decay': 0.000981287923867241, 
            'beta_0': 0.8141978952748745,
            'beta_1': 0.9905729096966865,
        },
        training_kw={
            'orient_loss_weight': 1.0,
            'demos_loss_weight': 0.6000000000000001,
        },
    )
    seq_metrics.to_parquet("seq_meta_data_metrics.parquet")
    print("saved sequence metrics data frame.")
    user_input = input("Upload model ensemble?").lower()
    if user_input == "yes":
        kagglehub.model_upload(
            handle=join(
                kagglehub.whoami()["username"],
                MODEL_NAME,
                "pyTorch",
                MODEL_VARIATION,
            ),
            local_model_dir="models",
            version_notes=input("Please provide model version notes:")
        )
    elif user_input == "no":
        print("Model has not been uploaded to kaggle.")
    else:
        print("User input was not understood, model has not been uploaded to kaggle.")