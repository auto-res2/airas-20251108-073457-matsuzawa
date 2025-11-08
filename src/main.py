import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ---------------- Mode handling ----------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.dataset.num_tasks = min(cfg.dataset.num_tasks, 2)
        cfg.training.task_epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be trial|full")

    # Resolve & persist config ------------------------------------------------
    out_dir = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / f"{cfg.run}_resolved.yaml")

    # Launch train subprocess -------------------------------------------------
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()