#!/usr/bin/env python3
"""Grid search for training tweaks (lr_rate, epochs) and evaluate each model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from typing import List, Tuple


def _parse_list(value: str, cast=float) -> List:
    if value is None or value.strip() == "":
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip() != ""]


def _list_run_dirs(outputs_dir: str) -> List[str]:
    if not os.path.exists(outputs_dir):
        return []
    run_dirs = []
    for name in os.listdir(outputs_dir):
        path = os.path.join(outputs_dir, name)
        if os.path.isdir(path):
            run_dirs.append(path)
    return run_dirs


def _latest_run_dir(before: List[str], after: List[str]) -> str | None:
    new_dirs = list(set(after) - set(before))
    if new_dirs:
        return max(new_dirs, key=os.path.getmtime)
    if after:
        return max(after, key=os.path.getmtime)
    return None


def _run_cmd(cmd: List[str], log_path: str) -> None:
    with open(log_path, "a") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, check=True, stdout=handle, stderr=handle)


def _load_report_scores(run_dir: str) -> Tuple[float, float, float]:
    reports = [f for f in os.listdir(run_dir) if f.startswith("Evaluation_report_") and f.endswith(".json")]
    if not reports:
        raise FileNotFoundError(f"missing Evaluation_report_*.json in {run_dir}")
    latest = max(reports, key=lambda name: os.path.getmtime(os.path.join(run_dir, name)))
    report_path = os.path.join(run_dir, latest)
    with open(report_path, "r") as handle:
        report = json.load(handle)
    overall = report.get("overall_scores", {})
    f1 = float(overall.get("fmeasure (percentage)", 0.0))
    precision = float(overall.get("precision", 0.0))
    recall = float(overall.get("recall", 0.0))
    return f1, precision, recall


def _tag_value(value: float) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_files_path", required=True)
    parser.add_argument("--team_name", default="TESTteam")
    parser.add_argument("--dataset", default="VAL")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--lrs", default="0.001,0.0005,0.0001")
    parser.add_argument("--epochs", default="5,10")
    args = parser.parse_args()

    lrs = _parse_list(args.lrs, float)
    epochs = _parse_list(args.epochs, int)
    if not lrs or not epochs:
        raise ValueError("lrs and epochs must be non-empty lists")

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)

    grid_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = os.path.join(outputs_dir, f"grid_train_{grid_stamp}")
    os.makedirs(grid_dir, exist_ok=True)
    run_outputs_dir = os.path.join(grid_dir, "runs")
    os.makedirs(run_outputs_dir, exist_ok=True)
    log_path = os.path.join(grid_dir, "grid.log")

    results = []
    total = len(lrs) * len(epochs)
    idx = 0

    for lr in lrs:
        for epoch in epochs:
            idx += 1
            model_tag = f"grid_train_{grid_stamp}_lr{_tag_value(lr)}_e{epoch}"

            before = _list_run_dirs(run_outputs_dir)
            cmd_train = [
                "python",
                "main.py",
                "set.features=false",
                "set.train=true",
                "set.eval=false",
                f"path.output_dir={run_outputs_dir}",
                f"path.model_tag={model_tag}",
                f"train.lr_rate={lr}",
                f"train.epochs={epoch}",
            ]
            print(f"[{idx}/{total}] lr={lr} epochs={epoch}")
            _run_cmd(cmd_train, log_path)
            after = _list_run_dirs(run_outputs_dir)
            train_run_dir = _latest_run_dir(before, after)
            if train_run_dir is None:
                raise RuntimeError("could not locate train run directory")

            before = _list_run_dirs(run_outputs_dir)
            cmd_eval = [
                "python",
                "main.py",
                "set.features=false",
                "set.train=false",
                "set.eval=true",
                f"path.output_dir={run_outputs_dir}",
                f"path.model_tag={model_tag}",
                "eval.sweep=false",
                f"eval.threshold={args.threshold}",
            ]
            _run_cmd(cmd_eval, log_path)
            after = _list_run_dirs(run_outputs_dir)
            eval_run_dir = _latest_run_dir(before, after)
            if eval_run_dir is None:
                raise RuntimeError("could not locate eval run directory")

            eval_cmd = [
                "python",
                "evaluation_metrics/evaluation.py",
                f"-pred_file={os.path.join(eval_run_dir, 'eval_output.csv')}",
                f"-ref_files_path={args.ref_files_path}",
                f"-team_name={args.team_name}",
                f"-dataset={args.dataset}",
                f"-savepath={eval_run_dir}",
            ]
            _run_cmd(eval_cmd, log_path)
            f1, precision, recall = _load_report_scores(eval_run_dir)
            results.append(
                {
                    "lr_rate": lr,
                    "epochs": epoch,
                    "threshold": args.threshold,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "model_tag": model_tag,
                    "train_run_dir": train_run_dir,
                    "eval_run_dir": eval_run_dir,
                }
            )
            time.sleep(0.1)

    results.sort(key=lambda x: x["f1"], reverse=True)
    csv_path = os.path.join(grid_dir, "grid_results.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    json_path = os.path.join(grid_dir, "grid_results.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)

    best = results[0]
    print("Grid summary")
    print(f"Best F1: {best['f1']}  threshold: {best['threshold']}")
    print(f"lr_rate={best['lr_rate']} epochs={best['epochs']}")
    print(f"Results saved to {grid_dir}")


if __name__ == "__main__":
    main()
