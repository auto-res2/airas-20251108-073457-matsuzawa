"""Independent evaluation & visualisation script.
Usage:
    uv run python -m src.evaluate results_dir=/path run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

# -----------------------------------------------------------------------------
# CLI parsing -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _parse_cli():
    results_dir, run_ids_json = None, None
    for arg in sys.argv[1:]:
        if arg.startswith("results_dir="):
            results_dir = arg.split("=", 1)[1]
        elif arg.startswith("run_ids="):
            run_ids_json = arg.split("=", 1)[1]
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
    if results_dir is None or run_ids_json is None:
        raise SystemExit("Usage: python -m src.evaluate results_dir=/path run_ids='[""run-1"", ...]' ")
    return Path(os.path.abspath(results_dir)), json.loads(run_ids_json)


# -----------------------------------------------------------------------------
# Helpers ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def export_metrics(history: "pd.DataFrame", summary: Dict, config: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": history.to_dict(orient="list"),
        "summary": summary,
        "config": config,
    }
    fp = out_dir / "metrics.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2)
    return fp


def save_learning_curve(history: "pd.DataFrame", run_id: str, out_dir: Path) -> Path:
    if "global_step" not in history.columns or "train_loss" not in history.columns:
        return Path()
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=history, x="global_step", y="train_loss", label="train_loss", color="tab:blue")
    if "val_retained_accuracy" in history.columns:
        ax2 = plt.twinx()
        sns.lineplot(
            data=history,
            x="global_step",
            y="val_retained_accuracy",
            label="val_retained_accuracy",
            ax=ax2,
            color="tab:orange",
        )
        ax2.set_ylabel("val_retained_accuracy")
    plt.title(f"Learning curve – {run_id}")
    plt.tight_layout()
    fp = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


def bar_chart(metric_map: Dict[str, float], title: str, out_path: Path):
    names, vals = zip(*sorted(metric_map.items(), key=lambda kv: kv[0]))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(names), y=list(vals))
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel(title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def box_violin(df: "pd.DataFrame", metric: str, out_dir: Path):
    gdf = df[df["metric"] == metric]
    if gdf.empty:
        return []
    paths = []
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=gdf, x="method", y="value")
    plt.title(f"{metric} – box-plot")
    plt.tight_layout()
    p = out_dir / f"comparison_{metric}_box_plot.pdf"
    plt.savefig(p)
    plt.close()
    paths.append(p)

    plt.figure(figsize=(8, 4))
    sns.violinplot(data=gdf, x="method", y="value", inner="point")
    plt.title(f"{metric} – violin-plot")
    plt.tight_layout()
    p2 = out_dir / f"comparison_{metric}_violin_plot.pdf"
    plt.savefig(p2)
    plt.close()
    paths.append(p2)
    return paths


# -----------------------------------------------------------------------------
# Main evaluation -------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    out_root, run_ids = _parse_cli()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_global = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    entity, project = cfg_global.wandb.entity, cfg_global.wandb.project

    api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = {}
    records: List[Dict] = []
    primary_metric_name = "retained_accuracy_auc"

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        hist = run.history()
        summ = dict(run.summary._json_dict)
        cfg = dict(run.config)

        run_dir = out_root / rid
        mpath = export_metrics(hist, summ, cfg, run_dir)
        lcurve_path = save_learning_curve(hist, rid, run_dir)
        print(mpath)
        if lcurve_path.exists():
            print(lcurve_path)

        for key, mat in summ.items():
            if key.startswith("confusion_matrix_"):
                task = key[len("confusion_matrix_") :]
                cm = np.array(mat)
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(cm, annot=False, cbar=False, ax=ax)
                plt.title(f"{rid} – {task}")
                plt.tight_layout()
                fp = run_dir / f"{rid}_{task}_confusion_matrix.pdf"
                fig.savefig(fp)
                plt.close(fig)
                print(fp)

        # Aggregate scalar summary metrics -----------------------------------
        for k, v in summ.items():
            if isinstance(v, (int, float)):
                aggregated.setdefault(k, {})[rid] = float(v)
                records.append(
                    {
                        "run_id": rid,
                        "metric": k,
                        "value": float(v),
                        "method": "proposed"
                        if "proposed" in rid
                        else "baseline"
                        if ("baseline" in rid or "comparative" in rid)
                        else "other",
                    }
                )

    # ---------------- Aggregated metrics ------------------------------------
    comp_dir = out_root / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    prim_map = aggregated.get(primary_metric_name, {})
    best_prop = max(
        ((rid, val) for rid, val in prim_map.items() if "proposed" in rid),
        key=lambda x: x[1],
        default=(None, -np.inf),
    )
    best_base = max(
        (
            (rid, val)
            for rid, val in prim_map.items()
            if ("baseline" in rid) or ("comparative" in rid)
        ),
        key=lambda x: x[1],
        default=(None, -np.inf),
    )
    gap = None
    if best_prop[1] != -np.inf and best_base[1] != -np.inf and best_base[1] != 0:
        gap = (best_prop[1] - best_base[1]) / abs(best_base[1]) * 100.0

    aggregate_json = comp_dir / "aggregated_metrics.json"
    with open(aggregate_json, "w") as f:
        json.dump(
            {
                "primary_metric": "Area-Under-Curve of ‘retained-accuracy vs tasks’ ↑; secondary – CPU latency overhead ↓",
                "metrics": aggregated,
                "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
                "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
                "gap": gap,
            },
            f,
            indent=2,
        )
    print(aggregate_json)

    # ---------------- Comparison figures -----------------------------------
    if prim_map:
        bar_path = comp_dir / "comparison_retained_accuracy_auc_bar_chart.pdf"
        bar_chart(prim_map, "Retained Accuracy AUC", bar_path)
        print(bar_path)

    df_records = pd.DataFrame(records)
    extra_paths = box_violin(df_records, primary_metric_name, comp_dir)
    for p in extra_paths:
        print(p)

    # ---------------- Statistical significance ----------------------------
    groups = {
        "proposed": df_records[
            (df_records.metric == primary_metric_name) & (df_records.method == "proposed")
        ].value.values,
        "baseline": df_records[
            (df_records.metric == primary_metric_name) & (df_records.method == "baseline")
        ].value.values,
    }
    if len(groups["proposed"]) >= 2 and len(groups["baseline"]) >= 2:
        t_stat, p_val = stats.ttest_ind(groups["proposed"], groups["baseline"], equal_var=False)
        tfile = comp_dir / "ttest.txt"
        with open(tfile, "w") as f:
            f.write(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}\n")
            f.write(
                f"proposed mean: {groups['proposed'].mean():.4f}, baseline mean: {groups['baseline'].mean():.4f}\n"
            )
        print(tfile)


if __name__ == "__main__":
    main()