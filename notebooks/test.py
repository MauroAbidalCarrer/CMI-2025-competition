import optuna
from optuna.pruners import BasePruner
from optuna.trial import TrialState
from optuna.exceptions import TrialPruned
from typing import Dict, List, Iterable, Any, Optional
import numpy as np
import random

class FoldPruner(BasePruner):
    def __init__(
        self,
        warmup_steps: int = 0,
        tolerance: float = 0.0,
    ):
        self.warmup_steps = int(warmup_steps)
        self.tolerance_abs = float(tolerance)

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        # trial.intermediate_values: dict step -> value
        intermidiate_vals = trial.intermediate_values
        if len(intermidiate_vals) <= self.warmup_steps:
            return False
        # use the most recent step (fold index) the trial reported
        current_step = max(intermidiate_vals.keys())
        # collect other trials' values at the same step
        other_vals = []
        for t in study.trials:
            if t.number != trial.number and current_step in t.intermediate_values:
                other_vals.append(t.intermediate_values[current_step])
        # compute reference value from other_vals
        ref = max(other_vals)
        current_val = intermidiate_vals[current_step]

        return current_val < (ref - self.tolerance_abs)

def train_on_all_folds(
        folds: Iterable[Any],
        run_fold_fn,
        trial_params: Dict[str, Any],
        trial: Optional[optuna.trial.Trial]=None,
    ) -> float:
    fold_scores = []
    for fold_idx, fold in enumerate(folds):
        # Train + evaluate a single fold. This function should be deterministic given seeding.
        val_score = run_fold_fn(fold_idx, fold, trial_params)

        fold_scores.append(val_score)

        if trial is not None:
            trial.report(val_score, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

    # Completed all folds -> return aggregate metric (mean here)
    # You may choose a different aggregation (median, weighted mean, etc.)
    return float(np.mean(fold_scores))


# --------------------------
# Example run_fold_fn (toy)
# --------------------------
def run_fold_fn_example(fold_idx: int, fold_info: Any, trial_params: Dict[str, Any]) -> float:
    """
    Replace this with your actual training/evaluation per fold.
    It must return a single float score (higher = better if study.direction=MAXIMIZE).
    Keep it deterministic (use seeding based on trial.number + fold_idx).
    """
    # Example: pseudo-random score influenced by params and fold_idx
    seed = int(trial_params.get("seed_base", 0)) + fold_idx + trial_params.get("seed_offset", 0)
    random.seed(seed)
    # pretend a param "max_lr" helps on easier folds but not on hard ones
    base = 0.6 + 0.1 * (trial_params.get("max_lr", 0.01) * 100)
    noise = random.gauss(0.0, 0.03)
    # simulate easier folds (higher index -> easier)
    difficulty = max(0.0, 1.0 - 0.05 * fold_idx)
    score = base * difficulty + noise
    return float(score)


# --------------------------
# Full objective and study example
# --------------------------
def objective(trial: optuna.trial.Trial, folds, run_fold_fn):
    # example of sampling hyper-parameters
    trial_params = {
        "max_lr": trial.suggest_float("max_lr", 0.001, 0.01, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "seed_base": 1234,
        "seed_offset": trial.number % 100,  # keep deterministic but varied
    }

    # train_on_all_folds will report intermediate values and raise TrialPruned if pruned
    mean_score = train_on_all_folds(
        trial=trial,
        folds=folds,
        run_fold_fn=run_fold_fn,
        trial_params=trial_params,
        report_step_offset=0,
        verbose=False,
    )
    return mean_score


if __name__ == "__main__":
    # Example usage with dummy folds and the toy run_fold_fn
    NB_FOLDS = 10
    folds = list(range(NB_FOLDS))  # replace with your actual folds (train/val splits)

    # Create custom pruner
    pruner = FoldPruner(
        min_trials=5,
        warmup_steps=0,
        comparator="max",        # choose from 'max'|'median'|'percentile'
        percentile=80.0,        # used if comparator == 'percentile'
        tolerance=0.01,     # absolute tolerance
        tolerance_rel=0.0,      # relative tolerance
        min_steps_reported=1,
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    # wrap objective so it receives folds and run_fold_fn
    func = lambda t: objective(t, folds=folds, run_fold_fn=run_fold_fn_example)

    study.optimize(func, n_trials=200)

    print("Study statistics:")
    pruned = study.get_trials(states=[TrialState.PRUNED])
    complete = study.get_trials(states=[TrialState.COMPLETE])
    print(f"  Trials: {len(study.trials)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Complete: {len(complete)}")

    best = study.best_trial
    print("Best trial value:", best.value)
    print("Best trial params:", best.params)
