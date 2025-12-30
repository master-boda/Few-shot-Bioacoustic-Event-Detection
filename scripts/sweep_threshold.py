"""Sweep detection thresholds on validation predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.metrics import Event, match_events, f_measure


def load_events(path: Path):
    df = pd.read_csv(path)
    start_col = "Starttime" if "Starttime" in df.columns else "Start"
    end_col = "Endtime" if "Endtime" in df.columns else "End"
    return [Event(float(s), float(e)) for s, e in zip(df[start_col], df[end_col])]


def main():
    parser = argparse.ArgumentParser(description="Sweep thresholds to maximize F-measure.")
    parser.add_argument("--pred_template", type=str, required=True, help="Prediction CSV per threshold (use {thr} placeholder).")
    parser.add_argument("--ref_csv", type=str, required=True, help="Reference CSV.")
    parser.add_argument("--ignore_before", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default="-1.0,-0.5,0.0,0.5,1.0")
    args = parser.parse_args()

    refs = [e for e in load_events(Path(args.ref_csv)) if e.end > args.ignore_before]
    best = None
    for t_str in args.thresholds.split(","):
        t = float(t_str)
        pred_path = Path(args.pred_template.format(thr=t))
        if not pred_path.exists():
            print(f"skip {pred_path} (missing)")
            continue
        preds = [e for e in load_events(pred_path) if e.end > args.ignore_before]
        tp, fn, fp = match_events(refs, preds, iou_thr=args.iou)
        f, p, r = f_measure(tp, fn, fp)
        print(f"thr={t} F={f:.4f} P={p:.4f} R={r:.4f} (tp={tp}, fn={fn}, fp={fp})")
        if best is None or f > best[0]:
            best = (f, t, p, r)
    if best:
        f, t, p, r = best
        print(f"Best: thr={t} F={f:.4f} P={p:.4f} R={r:.4f}")


if __name__ == "__main__":
    main()

