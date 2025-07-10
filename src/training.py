import os
# from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
import plotly.express as px
from rich.progress import Progress, Task, track
from pandas import DataFrame as DF
from torch.utils.data import DataLoader as DL
from torch.optim.lr_scheduler import LRScheduler


def fit(epochs:int,
        model: nn.Module,
        scheduler: LRScheduler,
        optimizer: torch.optim.Optimizer,
        train_loader: DL,
        criterion: callable=nn.L1Loss(),
        evaluation_func: callable=None,
        validation_loader: DL=None,
        save_checkpoints=True,
    ) -> DF:
    # Setup
    metrics: list[dict] = []
    checkpoints_dir = os.path.join("checkpoints", mk_date_now_str())
    step = 0
    # Training loop
    for epoch in range(epochs):
        total_epoch_loss = 0
        nb_samples = 0
        with Progress() as progress:
            task: Task = progress.add_task(
                f"epoch: {epoch + 1}, batch_loss: ...",
                total=len(train_loader) - 10,
            )
            for batch_idx, (x, y) in enumerate(train_loader):
                # forward
                x = x.cuda()
                y = y.cuda()
                nb_samples += len(x)
                model.train()
                optimizer.zero_grad()
                y_pred = model(x)
                loss_value = criterion(y_pred, y)
                # Verify loss value
                if torch.isnan(loss_value).any().item():
                    progress.print("Warning: Got NaN loss, something went wrong.")
                    return DF.from_records(metrics) 
                if torch.isinf(loss_value).any().item():
                    progress.print("Warning: Got infinite loss, something went wrong.")
                    return DF.from_records(metrics) 
                # TODO: Use gradient clipping?
                if step > 0: # If it's not the first training step
                    # Call the scheduler step method, idk why it throws an error otherwise
                    scheduler.step()
                # metrics
                total_epoch_loss += loss_value.item()
                metrics.append({
                    "step": step,
                    "epoch": epoch,
                    "batch_train_loss": loss_value.item(),
                    "lr": optimizer.state_dict()["param_groups"][-1]["lr"],
                })
                step += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"epoch: {epoch}, batch_loss: {(total_epoch_loss / (batch_idx+1)):.2f}"
                )
        # Post epoch evalution
        metrics[-1]["train_epoch_loss"] = total_epoch_loss / len(train_loader)
        if evaluation_func:
            eval_metrics = evaluation_func(model, criterion, validation_loader)
            progress.print("validation loss:", eval_metrics["validation_loss"])
            metrics[-1].update(eval_metrics)
        # Save checkpoint
        if save_checkpoints:
            checkpoint = mk_checkpoint(epoch, model, scheduler, optimizer)
            metrics_df = DF.from_records(metrics)
            best_model_metric = "validation_loss" if "validation_loss" in metrics_df.columns else "train_epoch_loss"
            is_best_checkpoint = (
                DF.from_records(metrics)
                .eval(f"min_{best_model_metric} = {best_model_metric}.min()")
                .eval(f"is_best_{best_model_metric} = {best_model_metric} == min_{best_model_metric}")
                .dropna(subset=f"is_best_{best_model_metric}")
                .iloc[-1]
                .loc[f"is_best_{best_model_metric}"]
            )
            save_checkpoint(checkpoint, checkpoints_dir, is_best_checkpoint)

    return DF.from_records(metrics)

def evaluate_model(model: torch.nn.Module, critirion:callable, validation_loader:DL) -> dict:
    model = model.eval()

    total_test_loss = 0
    for x, y in track(validation_loader, description="Evaluating...", transient=True):
        x = x.cuda()
        y = y.cuda()
        y_pred = model(x)
        total_test_loss += critirion(y_pred, y).item()
    total_test_loss /= len(validation_loader)

    return {"validation_loss": total_test_loss}

def mk_checkpoint(
        epoch:int,
        model: torch.nn.Module,
        scheduler: LRScheduler,
        optimizer: torch.optim.Optimizer
    ) -> dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

def save_checkpoint(checkpoint:dict, parent_dir:str, is_best_checkpoint=False):
    # Create model name
    checkpoint_filename = mk_date_now_str() + ".pth"
    # Save model
    os.makedirs(parent_dir, exist_ok=True)
    checkpoint_path = os.path.join(parent_dir, checkpoint_filename)
    torch.save(checkpoint, checkpoint_path)
    mk_symlink(checkpoint_path, os.path.join(parent_dir, "latest_checkpoint.pth"))
    if is_best_checkpoint:
        mk_symlink(checkpoint_path, os.path.join(parent_dir, "best_checkpoint.pth"))

def mk_date_now_str() -> str:
    return datetime.now().strftime("%d-%m-%Y_%H-%M")

def mk_symlink(dest:str, symlink_path:str):
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(dest, symlink_path)