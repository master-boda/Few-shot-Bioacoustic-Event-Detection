#!/usr/bin/env python3
"""Grid search for eval.test_hoplen_fenmu with fixed threshold."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from typing import List, Tuple


def _parse_list(value: str, cast=int) -> List:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_files_path", required=True)
    parser.add_argument("--team_name", default="TESTteam")
    parser.add_argument("--dataset", default="VAL")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--hop_factors", default="2,3,4")
    parser.add_argument("--samples_neg", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()

    hop_factors = _parse_list(args.hop_factors, int)
    if not hop_factors:
        raise ValueError("hop_factors must be a non-empty list")

    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)

    grid_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = os.path.join(outputs_dir, f"grid_hoplen_{grid_stamp}")
    os.makedirs(grid_dir, exist_ok=True)
    run_outputs_dir = os.path.join(grid_dir, "runs")
    os.makedirs(run_outputs_dir, exist_ok=True)
    log_path = os.path.join(grid_dir, "grid.log")

    results = []
    total = len(hop_factors)
    idx = 0

    for hop_factor in hop_factors:
        idx += 1
        before = _list_run_dirs(run_outputs_dir)
        tag = f"hop{hop_factor}"
        cmd = [
            "python",
            "main.py",
            "set.features=false",
            "set.train=false",
            "set.eval=true",
            f"path.output_dir={run_outputs_dir}",
            "eval.sweep=false",
            f"eval.threshold={args.threshold}",
            f"eval.samples_neg={args.samples_neg}",
            f"eval.iterations={args.iterations}",
            f"eval.test_hoplen_fenmu={hop_factor}",
            f"+eval.run_tag={tag}",
        ]

        print(f"[{idx}/{total}] test_hoplen_fenmu={hop_factor}")
        _run_cmd(cmd, log_path)
        after = _list_run_dirs(run_outputs_dir)
        run_dir = _latest_run_dir(before, after)
        if run_dir is None:
            raise RuntimeError("could not locate run directory")

        eval_cmd = [
            "python",
            "evaluation_metrics/evaluation.py",
            f"-pred_file={os.path.join(run_dir, 'eval_output.csv')}",
            f"-ref_files_path={args.ref_files_path}",
            f"-team_name={args.team_name}",
            f"-dataset={args.dataset}",
            f"-savepath={run_dir}",
        ]
        _run_cmd(eval_cmd, log_path)
        f1, precision, recall = _load_report_scores(run_dir)
        results.append(
            {
                "test_hoplen_fenmu": hop_factor,
                "threshold": args.threshold,
                "samples_neg": args.samples_neg,
                "iterations": args.iterations,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "run_dir": run_dir,
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
    print(f"test_hoplen_fenmu={best['test_hoplen_fenmu']}")
    print(f"Results saved to {grid_dir}")


if __name__ == "__main__":
    main()
