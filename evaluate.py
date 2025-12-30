"""Evaluate detections against annotations using a simplified event-based F-measure."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.official import load_events, match_events_hungarian, f_measure, fifth_pos_end


def main():
    parser = argparse.ArgumentParser(description="Event-based F-measure evaluation (Hungarian IoU matching)")
    parser.add_argument("--pred_csv", type=str, required=True, help="Predictions CSV with Starttime/Endtime")
    parser.add_argument("--ref_csv", type=str, required=True, help="Reference CSV with Starttime/Endtime/POS")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--ignore_before", type=float, default=None, help="Ignore events before this time; default uses 5th POS end")
    parser.add_argument("--drop_unk", action="store_true", help="Drop UNK rows from references")
    args = parser.parse_args()

    if args.ignore_before is None:
        ignore_before = fifth_pos_end(Path(args.ref_csv))
    else:
        ignore_before = args.ignore_before

    preds = [e for e in load_events(Path(args.pred_csv), drop_unk=False) if e.end > ignore_before]
    refs = [e for e in load_events(Path(args.ref_csv), drop_unk=args.drop_unk) if e.end > ignore_before]

    tp, fn, fp = match_events_hungarian(refs, preds, iou_thr=args.iou)
    f, p, r = f_measure(tp, fn, fp)
    print(f"Ignore_before={ignore_before:.3f} | F={f:.4f} P={p:.4f} R={r:.4f} (tp={tp}, fn={fn}, fp={fp})")


if __name__ == "__main__":
    main()
