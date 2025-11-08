import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import src.model as model_lib
import src.preprocess as prep_lib

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def _safe_len(loader: DataLoader) -> Optional[int]:
    try:
        return len(loader)
    except TypeError:
        return None


# -----------------------------------------------------------------------------
# Fisher information estimation (for EWC on LILAC baseline) --------------------
# -----------------------------------------------------------------------------

def compute_fisher(model: nn.Module, dataloader: DataLoader, device: torch.device, samples: int = 128):
    """Diagonal Fisher estimation (light-weight, replay-free)."""
    fisher: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, dtype=torch.float32, device=device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    model.eval()
    processed = 0
    for batch in dataloader:
        if processed >= samples:
            break
        processed += 1
        batch = _move_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    for name in fisher:
        fisher[name] /= max(processed, 1)
    return fisher


# -----------------------------------------------------------------------------
# Validation helpers -----------------------------------------------------------
# -----------------------------------------------------------------------------

def evaluate_stream(
    model: nn.Module,
    loaders: List[Tuple[str, DataLoader]],
    device: torch.device,
    cfg: DictConfig,
    global_step: int,
):
    model.eval()
    per_task_loss = {}
    retained_scores: List[float] = []
    conf_mats: Dict[str, List[List[int]]] = {}

    with torch.no_grad():
        for task_name, loader in loaders:
            total_tok, total_loss = 0, 0.0
            y_true, y_pred = [], []
            for batch_idx, batch in enumerate(loader):
                if cfg.mode == "trial" and batch_idx > 1:
                    break
                batch = _move_to_device(batch, device)
                outputs = model(**batch)
                loss = outputs["loss"].detach()
                total_loss += loss.item() * batch["input_ids"].numel()
                total_tok += batch["input_ids"].numel()

                # For token-level generative loss we simply compute retained-accuracy proxy
                if outputs["logits"].dim() == 3:
                    continue

                preds = outputs["logits"].argmax(dim=-1).flatten().cpu()
                labels = batch["labels"].flatten().cpu()
                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())
            mean_loss = total_loss / max(total_tok, 1)
            per_task_loss[task_name] = mean_loss
            retained_scores.append(1.0 / (1.0 + mean_loss))  # proxy ↑ if loss ↓
            if y_true:
                cm = confusion_matrix(y_true, y_pred)
                conf_mats[task_name] = cm.tolist()

    metrics = {
        "val_retained_accuracy": float(np.mean(retained_scores)) if retained_scores else 0.0,
        **{f"val_loss_{k}": float(v) for k, v in per_task_loss.items()},
    }
    return metrics, conf_mats


# -----------------------------------------------------------------------------
# Raspberry-Pi-4 latency (CPU) --------------------------------------------------
# -----------------------------------------------------------------------------

def measure_cpu_latency(model: nn.Module, tokenizer, seq_len: int = 64, trials: int = 25):
    model_cpu = model.to("cpu", dtype=torch.float32).eval()
    dummy = torch.randint(0, tokenizer.vocab_size, (1, seq_len), dtype=torch.long)
    attn = torch.ones_like(dummy)
    with torch.no_grad():
        for _ in range(5):  # warm-up
            model_cpu(input_ids=dummy, attention_mask=attn)
        start = time.time()
        for _ in range(trials):
            model_cpu(input_ids=dummy, attention_mask=attn)
        end = time.time()
    return (end - start) / trials * 1000.0  # ms


# -----------------------------------------------------------------------------
# Core training loop -----------------------------------------------------------
# -----------------------------------------------------------------------------

def train_full_stream(cfg: DictConfig, trial: Optional[optuna.trial.Trial] = None):
    """Train one continual task stream.  If `trial` is not None we are inside
    an Optuna optimisation loop and *must not* log anything to WandB."""

    use_wandb = (trial is None) and (cfg.wandb.mode != "disabled")
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    import wandb  # local import respects WANDB_MODE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Model & Data ----------------------------------------
    model, tokenizer = model_lib.build_model(cfg)
    model.to(device)
    task_stream = list(prep_lib.get_task_stream(cfg, tokenizer))

    # ---------------- Optimiser ------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = GradScaler('cuda', enabled=cfg.training.mixed_precision and torch.cuda.is_available())

    # ---------------- WandB init -----------------------------------------
    if use_wandb:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print("WandB URL:", wandb.run.url)

    fisher_prev: Dict[str, torch.Tensor] = {}
    opt_prev: Dict[str, torch.Tensor] = {}
    global_step = 0
    retained_curve: List[float] = []

    # ---------------- Training over tasks --------------------------------
    for task_idx, (task_name, train_loader, val_loader) in enumerate(task_stream, 1):
        print(f"\n=== Task {task_idx}/{cfg.dataset.num_tasks}: {task_name} ===")
        steps_per_epoch = _safe_len(train_loader) or cfg.training.max_updates_per_task // cfg.training.task_epochs
        total_updates = (
            steps_per_epoch * cfg.training.task_epochs if steps_per_epoch else cfg.training.max_updates_per_task
        )
        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=min(cfg.training.warmup_steps, max(total_updates // 10, 1)),
                num_training_steps=total_updates,
            )
            if steps_per_epoch
            else None
        )

        for epoch in range(cfg.training.task_epochs):
            model.train()
            pbar_it = enumerate(train_loader)
            if steps_per_epoch:
                pbar_it = tqdm(pbar_it, total=steps_per_epoch, desc=f"{task_name} e{epoch}")
            for step, batch in pbar_it:
                if cfg.mode == "trial" and step > 1:
                    break
                batch = _move_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=cfg.training.mixed_precision and torch.cuda.is_available()):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    # EWC regulariser for LILAC baseline -------------------
                    if cfg.model.adapter.type.upper() == "LILAC" and fisher_prev:
                        ewc = 0.0
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in fisher_prev:
                                ewc += (fisher_prev[n] * (p - opt_prev[n]).pow(2)).sum()
                        loss = loss + cfg.model.adapter.get("ewc_lambda", 0.0) * ewc
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "lr": optimizer.param_groups[0]["lr"],
                            "task": task_idx,
                            "global_step": global_step,
                            "epoch": epoch,
                        }
                    )

        # -------- Validation ---------------------------------------------
        metrics, confmats = evaluate_stream(model, [(task_name, val_loader)], device, cfg, global_step)
        retained_curve.append(metrics["val_retained_accuracy"])
        if use_wandb:
            wandb.log({**metrics, "task": task_idx, "global_step": global_step})
            for name, cm in confmats.items():
                wandb.summary[f"confusion_matrix_{name}"] = cm

        # -------- Fisher info (for LILAC + EWC) ---------------------------
        if cfg.model.adapter.type.upper() == "LILAC":
            fisher_prev = compute_fisher(model, val_loader, device)
            opt_prev = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }

    # ---------------- End-of-stream metrics ------------------------------
    auc = 0.0
    for i in range(1, len(retained_curve)):
        auc += 0.5 * (retained_curve[i] + retained_curve[i - 1])
    cpu_lat = (
        measure_cpu_latency(model, tokenizer)
        if cfg.evaluation.cpu_latency_device == "raspberry-pi-4"
        else 0.0
    )

    if use_wandb:
        wandb.summary["retained_accuracy_auc"] = auc
        wandb.summary["cpu_latency_overhead_ms"] = cpu_lat
        wandb.finish()

    # Optuna objective value
    if trial is not None:
        trial.set_user_attr("auc", auc)
        return auc
    return auc


# -----------------------------------------------------------------------------
# Optuna hyper-parameter search ------------------------------------------------
# -----------------------------------------------------------------------------

def launch_optuna(cfg: DictConfig):
    """Run Optuna and return best hyper-parameters as dict."""

    def _objective(trial: optuna.Trial):
        cfg_t = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        # Apply sampled params -------------------------------------------------
        for hp, space in cfg.optuna.search_space.items():
            if space["type"] == "loguniform":
                val = trial.suggest_float(hp, space["low"], space["high"], log=True)
            elif space["type"] == "uniform":
                val = trial.suggest_float(hp, space["low"], space["high"])
            elif space["type"] == "categorical":
                val = trial.suggest_categorical(hp, space["choices"])
            else:
                raise ValueError(space["type"])
            OmegaConf.update(cfg_t, hp, val, merge=True)

        # short stream for tuning
        original_tasks = cfg_t.dataset.num_tasks
        cfg_t.dataset.num_tasks = min(3, original_tasks)
        score = train_full_stream(cfg_t, trial)
        cfg_t.dataset.num_tasks = original_tasks
        return score

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(_objective, n_trials=cfg.optuna.n_trials)
    return study.best_params


# -----------------------------------------------------------------------------
# Hydra entry-point -----------------------------------------------------------
# -----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Mode handling -----------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.dataset.num_tasks = min(cfg.dataset.num_tasks, 2)
        cfg.training.task_epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    set_seed()

    # Hyper-parameter search --------------------------------------------------
    if cfg.optuna.n_trials > 0:
        best_params = launch_optuna(cfg)
        for k, v in best_params.items():
            OmegaConf.update(cfg, k, v, merge=True)
        print("[Optuna] best params applied:", best_params)

    # Main training -----------------------------------------------------------
    train_full_stream(cfg)


if __name__ == "__main__":
    main()