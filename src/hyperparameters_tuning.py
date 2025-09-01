from typing import Optional
from functools import partial

import torch
import optuna
from pandas import DataFrame as DF
from optuna.trial import TrialState
from optuna.pruners import BasePruner
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as DL

from config import *
from model import mk_model_ensemble
from preprocessing import get_meta_data
from dataset import split_dataset, get_fold_datasets, move_cmi_dataset
from training import train_on_all_folds, evaluate_model

class FoldPruner(BasePruner):
    def __init__(
        self,
        warmup_steps: int = 0,
        tolerance: float = 0.0,
    ):
        self.warmup_steps = int(warmup_steps)
        self.tolerance = float(tolerance)

    def prune(self, study: optuna.study.Study, current_trial: optuna.trial.FrozenTrial) -> bool:
        nb_trials = len(study.trials)
        if nb_trials < 2 or nb_trials < self.warmup_steps:
            return False
        
        current_fold = max(current_trial.intermediate_values.keys())
        # collect other trials' values at the same fold
        other_vals = []
        for t in study.trials[:-1]:
            if current_fold in t.intermediate_values:
                other_vals.append(t.intermediate_values[current_fold])
        # compute reference value from other_vals
        ref = max(other_vals)
        current_val = current_trial.intermediate_values[current_fold]

        return current_val < (ref - self.tolerance)

def suggest_offseted_hp(
        suggest_fn: callable,
        val_name: str,
        dflt_dict: dict,
        offset:float|int,
        step:Optional[float|int]
    ) -> float|int:
    return suggest_fn(
        name=val_name,
        low=max(dflt_dict[val_name] - offset, 0),
        high=dflt_dict[val_name] + offset,
        step=step,
    )

def objective(
        trial: optuna.trial.Trial,
        preprocessed_meta_data: dict,
        val_loader: DL,
        val_device: torch.device,
    ) -> float:
    train_kw = {
        "orient_loss_weight": suggest_offseted_hp(
            trial.suggest_float,
            "orient_loss_weight",
            DFLT_TRAINING_HP_KW,
            offset=0.15,
            step=0.1,
        ),
        "sex_loss_weight": suggest_offseted_hp(
            trial.suggest_float,
            "sex_loss_weight",
            DFLT_TRAINING_HP_KW,
            offset=0.2,
            step=0.1,
        ),
        "handedness_loss_weight": suggest_offseted_hp(
            trial.suggest_float,
            "handedness_loss_weight",
            DFLT_TRAINING_HP_KW,
            offset=0.2,
            step=0.1,
        )
    }
    lr_scheduler_kw = {
        'warmup_epochs': trial.suggest_int("warmup_epochs", 14, 18),
        "cycle_mult": trial.suggest_float("cycle_mult", 0.8, 1.6, step=0.1),
        "init_cycle_epochs": trial.suggest_int("init_cycle_epochs", 2, 10),
        "max_lr": suggest_offseted_hp(
            trial.suggest_float,
            "max_lr",
            DFLT_LR_SCHEDULER_HP_KW,
            0.001,
            step=0.0001
        ),
        #trial.suggest_float("lr_cycle_factor", 0.25, 0.6, step=0.05),
    }
    model_kw = {"group_thm_branch": trial.suggest_categorical("group_thm_branch", [False, True])}

    train_on_all_folds(
        "train",
        training_kw=DFLT_TRAINING_HP_KW | train_kw,
        lr_scheduler_kw=DFLT_LR_SCHEDULER_HP_KW | lr_scheduler_kw,
        optimizer_kw={
            'weight_decay': trial.suggest_float("weight_decay", 0.000901287923867241, 0.001001287923867241, step=0.00001), 
            'beta_0': trial.suggest_float("beta_0", 0.8101978952748745, 0.8201978952748745, step=0.001),
            'beta_1': trial.suggest_float("beta_1", 0.9855729096966865, 0.9955729096966865, step=0.001),
        },
        model_kw=model_kw,
    )
    ensemble = mk_model_ensemble("models", val_device, model_kw)
    val_metrics = evaluate_model(preprocessed_meta_data, ensemble, val_loader, torch.nn.CrossEntropyLoss(), val_device)
    return val_metrics["final_metric"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=FoldPruner(warmup_steps=0, tolerance=0.002))
    splits = split_dataset()
    val_device = torch.device("cuda")
    val_dataset = move_cmi_dataset(splits["validation"][0], val_device)
    val_loader = DL(val_dataset, VALIDATION_BATCH_SIZE, shuffle=True)
    preprocessed_meta_data = get_meta_data()
    part_objective = partial(
        objective,
        val_loader=val_loader,
        preprocessed_meta_data=preprocessed_meta_data,
        val_device=val_device,
    )
    study.optimize(part_objective, n_trials=100, timeout=60 * 60 * 8)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    current_trial = study.best_trial

    print("  Value: ", current_trial.value)
    print("  Params: ")
    for key, value in current_trial.params.items():
        print("    {}: {}".format(key, value))