#!/usr/bin/env python3
"""Grid search for transductive inference settings with sweep evaluation.

Writes a grid summary (CSV/JSON) plus per-run logs to an outputs subfolder.
"""

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


def _load_best_from_sweep(run_dir: str) -> Tuple[float, float, float]:
    sweep_path = os.path.join(run_dir, "sweep_scores.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"missing sweep_scores.json in {run_dir}")
    with open(sweep_path, "r") as handle:
        scores = json.load(handle)
    best = max(scores, key=lambda x: x["fmeasure"])
    return float(best["threshold"]), float(best["fmeasure"]), float(best["precision"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_files_path", required=True)
    parser.add_argument("--team_name", default="TESTteam")
    parser.add_argument("--dataset", default="VAL")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--sweep_start", type=float, default=0.4)
    parser.add_argument("--sweep_stop", type=float, default=0.8)
    parser.add_argument("--sweep_step", type=float, default=0.05)
    parser.add_argument("--steps", default="1,2,3")
    parser.add_argument("--temps", default="1.0,2.0,3.0")
    parser.add_argument("--weights", default="0.05,0.1,0.2")
    parser.add_argument("--postprocess", default=None, choices=["true", "false"])
    parser.add_argument("--min_duration_frac", type=float, default=None)
    args = parser.parse_args()

    steps = _parse_list(args.steps, int)
    temps = _parse_list(args.temps, float)
    weights = _parse_list(args.weights, float)
    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)

    grid_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_dir = os.path.join(outputs_dir, f"grid_{grid_stamp}")
    os.makedirs(grid_dir, exist_ok=True)
    run_outputs_dir = os.path.join(grid_dir, "runs")
    os.makedirs(run_outputs_dir, exist_ok=True)
    log_path = os.path.join(grid_dir, "grid.log")

    results = []
    total = len(steps) * len(temps) * len(weights)
    idx = 0

    for step in steps:
        for temp in temps:
            for weight in weights:
                idx += 1
                before = _list_run_dirs(run_outputs_dir)
                cmd = [
                    "python",
                    "main.py",
                    "set.features=false",
                    "set.train=false",
                    "set.eval=true",
                    f"path.output_dir={run_outputs_dir}",
                    f"eval.transductive=true",
                    f"eval.transductive_steps={step}",
                    f"eval.transductive_temp={temp}",
                    f"eval.transductive_query_weight={weight}",
                    f"+eval.run_tag=td_s{step}_t{temp}_w{weight}",
                    "eval.sweep=true",
                    f"eval.sweep_start={args.sweep_start}",
                    f"eval.sweep_stop={args.sweep_stop}",
                    f"eval.sweep_step={args.sweep_step}",
                ]
                if args.postprocess is not None:
                    cmd.append(f"eval.postprocess={args.postprocess}")
                if args.min_duration_frac is not None:
                    cmd.append(f"eval.min_duration_frac={args.min_duration_frac}")

                print(f"[{idx}/{total}] steps={step} temp={temp} weight={weight}")
                _run_cmd(cmd, log_path)
                after = _list_run_dirs(run_outputs_dir)
                run_dir = _latest_run_dir(before, after)
                if run_dir is None:
                    raise RuntimeError("could not locate run directory")

                eval_cmd = [
                    "python",
                    "scripts/eval_sweep.py",
                    f"--sweep_dir={run_dir}",
                    f"--ref_files_path={args.ref_files_path}",
                    f"--team_name={args.team_name}",
                    f"--dataset={args.dataset}",
                ]
                _run_cmd(eval_cmd, log_path)
                best_thresh, best_f1, best_precision = _load_best_from_sweep(run_dir)
                results.append(
                    {
                        "steps": step,
                        "temp": temp,
                        "weight": weight,
                        "best_threshold": best_thresh,
                        "best_f1": best_f1,
                        "best_precision": best_precision,
                        "run_dir": run_dir,
                    }
                )
                time.sleep(0.1)

    results.sort(key=lambda x: x["best_f1"], reverse=True)
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
    print(f"Best F1: {best['best_f1']}  threshold: {best['best_threshold']}")
    print(f"steps={best['steps']} temp={best['temp']} weight={best['weight']}")
    print(f"Results saved to {grid_dir}")


if __name__ == "__main__":
    main()
