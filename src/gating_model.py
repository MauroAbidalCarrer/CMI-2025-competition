from os.path import join
from functools import partial
from tqdm.notebook import tqdm
from collections import defaultdict
from typing import Optional, Literal
from itertools import pairwise, starmap, product

import torch
import kagglehub 
import numpy as np
import pandas as pd
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from pandas import DataFrame as DF
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from config import *
from model import mk_model
from utils import seed_everything
from preprocessing import get_meta_data
from training import CMIDataset, sgkf_cmi_dataset


def record_model_outputs(model:nn.Module, data_loader:DL, device:torch.device) -> tuple[Tensor]:
    data:list[tuple[Tensor]] = []
    model = model.eval()
    tof_and_thm_idx = np.concatenate((meta_data["tof_idx"], meta_data["thm_idx"]))
    with torch.no_grad():
        for x, *_ in data_loader:
            x = x.to(device).clone()
            x[:1024 // 2, tof_and_thm_idx] = 0.0
            data.append(model(x))
    data:tuple[Tensor] = tuple(map(torch.concat, zip(*data)))
    return data

def mk_gating_model_dataset() -> TensorDataset:
    device = torch.device("cuda")
    dataset = CMIDataset(device)
    data_loader = DL(dataset, batch_size=1024, shuffle=False)
    models_outputs: list[tuple[Tensor]] = []
    for fold_idx in tqdm(range(N_FOLDS), total=N_FOLDS):
        model = mk_model(device=device)
        checkpoint = torch.load(
            join(
                "models",
                f"model_fold_{fold_idx}.pth"
            ),
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        models_outputs.append(record_model_outputs(model, data_loader, device))
    models_outputs: tuple[Tensor] = tuple(map(partial(torch.stack, dim=1), zip(*models_outputs)))
    tensors = (*models_outputs, *dataset.tensors[-2:], dataset.tensors[1])
    
    return TensorDataset(*tensors)

def evaluate_gating_model(dataset: Dataset, gating_model: nn.Module) -> dict:
    gating_model = gating_model.eval()
    data_loader = DL(dataset, batch_size=1024, shuffle=False)
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for *gating_inputs, y in data_loader:
            y_preds.append(gating_model(*gating_inputs))
            y_trues.append(y)
    y_pred = torch.argmax(torch.concat(y_preds), dim=1).cpu().numpy()
    y_true = torch.argmax(torch.concat(y_trues), dim=1).cpu().numpy()
    model_is_true = y_pred == y_true
    binary_true = np.isin(y_true, BFRB_INDICES).astype(int)
    binary_pred = np.isin(y_pred, BFRB_INDICES).astype(int)
    metrics = {
        "accuracy": model_is_true.mean().item(),
        "binary_f1": f1_score(binary_true, binary_pred),
    }

    # Collapse non-BFRB gestures into a single class
    collapsed_true = np.where(
        np.isin(y_true, BFRB_INDICES),
        y_true,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )
    collapsed_pred = np.where(
        np.isin(y_pred, BFRB_INDICES),
        y_pred,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )

    # Macro F1 on collapsed classes
    metrics["macro_f1"] = f1_score(collapsed_true, collapsed_pred, average='macro')
    metrics["final_metric"] = (metrics["binary_f1"] + metrics["macro_f1"]) / 2

    return metrics

def train_model_on_single_epoch(data_loader: DL, gating_model: nn.Module, criterion: nn.Module, optimizer: Optimizer) -> dict:
    metrics = defaultdict(float)
    n_samples = 0
    gating_model = gating_model.train()
    for *gating_inputs, y_true in data_loader:
        optimizer.zero_grad()
        y_pred = gating_model(*gating_inputs)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        n_samples += gating_inputs[0].shape[0]
        metrics["train_loss"] += loss.item() * gating_inputs[0].shape[0]

    metrics["train_loss"] /= n_samples

    return metrics

def train_model_on_all_epochs(dataset: Dataset, gating_model: nn.Module) -> DF:
    train_loader = DL(dataset, GATING_MODEL_TRAIN_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(gating_model.parameters())
    metrics: list[dict] = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(N_GATING_MODEL_EPOCHS):
        train_metrics = train_model_on_single_epoch(train_loader, gating_model, criterion, optimizer)
        eval_metrics = evaluate_gating_model(dataset, gating_model)
        metrics.append({"epoch": epoch} | train_metrics | eval_metrics)

    return DF.from_records(metrics)

class MeanGate(nn.Module):
    def ___init__(self, device: torch.device):
        self.device = device

    def forward(self,
            y_preds: Tensor,
            orient_preds: Tensor,
            bin_demos_y_preds: Tensor,
            reg_demos_y_preds: Tensor,
            bin_demos_y_true: Tensor,
            reg_demos_y_true: Tensor
        ) -> Tensor:
        return y_preds.mean(dim=1)

def preds_uncertainty(y_preds: Tensor) -> Tensor:
    """
    y_preds: Tensor[batch, n_folds, y_targets]
    returns: Tensor[batch, n_folds]
    """
    clipped_preds = y_preds.clip(EPSILON, 1.0)
    return  -((clipped_preds * torch.log(clipped_preds)).sum(dim=2))

def mae(y_preds: Tensor, y_true: Tensor) -> Tensor:
    """
    y_preds: Tensor[batch, n_folds, y_targets]
    y_true:  Tensor[batch, y_targets]
    returns: Tensor[batch, n_folds]
    """
    return torch.abs(y_preds - y_true.unsqueeze(1)).mean(dim=2)

class CMIGatingModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 64
        self.gate = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_FOLDS),
            nn.Sigmoid(),
        )

    def forward(
            self,
            y_preds: Tensor,
            orient_preds: Tensor,
            bin_demos_y_preds: Tensor,
            reg_demos_y_preds: Tensor,
            bin_demos_y_true: Tensor,
            reg_demos_y_true: Tensor
        ) -> Tensor:      
        
        experts_preds_stats = torch.concatenate((
                preds_uncertainty(y_preds),
                preds_uncertainty(orient_preds),
                preds_uncertainty(bin_demos_y_preds),
                preds_uncertainty(reg_demos_y_preds),
                mae(bin_demos_y_preds, bin_demos_y_true),
                mae(reg_demos_y_preds, reg_demos_y_true),
            ),
            dim=1,
        )
        weights = self.gate(experts_preds_stats)
        weighted_y_preds = torch.einsum("be, bet -> bt", weights, y_preds)
        y_pred = weighted_y_preds / weights.sum(dim=1, keepdim=True)

        return y_pred

def train_and_eval_gating_model(dataset: Dataset, device: torch.device) -> tuple[nn.Module, DF]:
    mean_gate_metrics = evaluate_gating_model(dataset, MeanGate())
    print("mean_gate_metrics:", mean_gate_metrics)
    gating_model = CMIGatingModel().to(device)
    base_logistic_gate_metrics = evaluate_gating_model(dataset, gating_model)
    print("base logistic_gate_metrics:", base_logistic_gate_metrics)
    training_metrics = train_model_on_all_epochs(dataset, gating_model)
    print("trained logistic_gate_metrics:", training_metrics.iloc[-1].to_dict())

    return gating_model, training_metrics


if __name__ == "__main__":
    seed_everything(SEED)
    meta_data = get_meta_data()
    dataset = mk_gating_model_dataset()
    device = torch.device("cuda")
    for fold_idx, (train_idx, validation_idx, seed) in enumerate(sgkf_cmi_dataset(dataset, n_splits=5)):
        print("fold", fold_idx)
        train_dataset = Subset(dataset, train_idx)
        validation_dataset = Subset(dataset, validation_idx)
        gating_model, metrics = train_and_eval_gating_model(train_dataset, device)
        print("trained test eval:", evaluate_gating_model(validation_dataset, gating_model))
        print("==========")

    torch.save(gating_model.state_dict(), f"models/gating_model.pth")
    user_input = input("Upload model ensemble?: ").lower()
    if user_input == "yes":
        kagglehub.model_upload(
            handle=join(
                kagglehub.whoami()["username"],
                MODEL_NAME,
                "pyTorch",
                MODEL_VARIATION,
            ),
            local_model_dir="models",
            version_notes=input("Please provide model version notes: ")
        )
    elif user_input == "no":
        print("Model has not been uploaded to kaggle.")
    else:
        print("User input was not understood, model has not been uploaded to kaggle.")