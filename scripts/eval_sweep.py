#!/usr/bin/env python3
"""Evaluate a sweep of prediction files and report the best threshold."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from typing import Dict, Tuple


def _extract_threshold(path: str) -> float | None:
    name = os.path.basename(path)
    match = re.search(r"eval_output_thresh_([0-9]+p[0-9]+)\.csv$", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def _run_eval(eval_script: str, pred_file: str, ref_path: str, team: str, dataset: str, out_dir: str) -> Dict:
    report_glob = os.path.join(out_dir, f"Evaluation_report_{team}_{dataset}_*.json")
    before_reports = set(glob.glob(report_glob))
    cmd = [
        "python",
        eval_script,
        f"-pred_file={pred_file}",
        f"-ref_files_path={ref_path}",
        f"-team_name={team}",
        f"-dataset={dataset}",
        f"-savepath={out_dir}",
    ]
    subprocess.run(cmd, check=True)
    after_reports = set(glob.glob(report_glob))
    new_reports = list(after_reports - before_reports)
    if new_reports:
        report_path = max(new_reports, key=os.path.getmtime)
    else:
        candidates = list(after_reports)
        if not candidates:
            raise FileNotFoundError(f"no evaluation report found in {out_dir}")
        report_path = max(candidates, key=os.path.getmtime)
    with open(report_path, "r") as handle:
        return json.load(handle)


def _scores_from_report(report: Dict) -> Tuple[float, float, float]:
    overall = report.get("overall_scores", report)
    precision = float(overall.get("precision"))
    recall = float(overall.get("recall"))
    f1 = None
    if "fmeasure (percentage)" in overall:
        f1 = float(overall["fmeasure (percentage)"]) / 100.0
    elif "fmeasure" in overall:
        f1 = float(overall["fmeasure"])
    if f1 is None:
        raise KeyError("could not find fmeasure in report")
    return precision, recall, f1


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "fmeasure", "file"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_pr(scores: list[dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PR plot (matplotlib unavailable): {exc}")
        return

    scores_sorted = sorted(scores, key=lambda x: x["threshold"])
    recall = [row["recall"] for row in scores_sorted]
    precision = [row["precision"] for row in scores_sorted]

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path = os.path.join(out_dir, "precision_recall_curve.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"PR curve saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True, help="Directory containing eval_output_thresh_*.csv files")
    parser.add_argument("--ref_files_path", required=True, help="Path to validation set folder")
    parser.add_argument("--team_name", default="TESTteam")
    parser.add_argument("--dataset", default="VAL")
    parser.add_argument("--eval_script", default="evaluation_metrics/evaluation.py")
    parser.add_argument("--out_dir", default=None, help="Where to write eval reports (default: sweep_dir)")
    parser.add_argument("--plot_pr", action="store_true", help="Generate precision-recall curve plot")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    out_dir = args.out_dir or sweep_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(sweep_dir, "eval_output_thresh_*.csv")))
    if not files:
        raise FileNotFoundError(f"no sweep files found in {sweep_dir}")

    best: Tuple[float, float, str] | None = None
    all_scores = []
    for pred in files:
        thresh = _extract_threshold(pred)
        if thresh is None:
            continue
        report = _run_eval(args.eval_script, pred, args.ref_files_path, args.team_name, args.dataset, out_dir)
        precision, recall, f1 = _scores_from_report(report)
        all_scores.append(
            {
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "fmeasure": f1,
                "file": pred,
            }
        )
        if best is None or f1 > best[1]:
            best = (thresh, f1, pred)

    all_scores.sort(key=lambda x: x["threshold"])
    out_path = os.path.join(out_dir, "sweep_scores.json")
    with open(out_path, "w") as handle:
        json.dump(all_scores, handle, indent=2)
    _write_csv(os.path.join(out_dir, "sweep_scores.csv"), all_scores)

    if best is None:
        raise RuntimeError("no valid sweep scores computed")

    f1_values = [row["fmeasure"] for row in all_scores]
    max_f1 = max(f1_values)
    min_f1 = min(f1_values)
    top_sorted = sorted(all_scores, key=lambda x: x["fmeasure"], reverse=True)[:3]
    top_str = ", ".join([f"{row['threshold']}: {row['fmeasure']}" for row in top_sorted])

    print("Sweep summary")
    print(f"Thresholds evaluated: {len(all_scores)}")
    print(f"Best threshold: {best[0]}  F1: {best[1]}  file: {best[2]}")
    print(f"F1 range: {min_f1} - {max_f1}")
    print(f"Top 3 thresholds: {top_str}")
    print(f"All scores written to {out_path}")
    if args.plot_pr:
        _plot_pr(all_scores, out_dir)


if __name__ == "__main__":
    main()
