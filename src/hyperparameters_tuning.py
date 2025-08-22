import optuna
from optuna.trial import TrialState
from optuna.pruners import BasePruner

from config import *

from training import train_on_all_folds

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

def objective(trial: optuna.trial.Trial) -> float:
    return train_on_all_folds(
        lr_scheduler_kw={
            "warmup_epochs": trial.suggest_int("warmup_epochs", 12, 15),
            "cycle_mult": trial.suggest_float("cycle_mult", 0.9, 1.6, step=0.1),
            "max_lr": trial.suggest_float("max_lr", 0.005581907927062619 / 1.5, 0.005581907927062619 * 1.5, step=0.0001),
            "max_to_min_div_factor": 250, #trial.suggest_float("max_to_min_div_factor", 100, 300, step=25),
            "init_cycle_epochs": trial.suggest_int("init_cycle_epochs", 2, 10, ),
            "lr_cycle_factor": trial.suggest_float("lr_cycle_factor", 0.25, 0.6, step=0.05),
        },
        optimizer_kw={
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-3),
            "beta_0":trial.suggest_float("beta_0", 0.8, 0.999),
            "beta_1":trial.suggest_float("beta_1", 0.99, 0.9999),
        },
        training_kw={
            "orient_loss_weight": trial.suggest_float("orient_loss_weight", 0, 1, step=0.1),
            "demos_loss_weight": trial.suggest_float("demos_loss_weight", 0, 1, step=0.1),
        },
        trial=trial,
    )[0]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=FoldPruner(warmup_steps=0, tolerance=0.002))
    study.optimize(objective, n_trials=100, timeout=60 * 60 * 3)

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