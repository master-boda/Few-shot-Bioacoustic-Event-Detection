"""Event-based F-measure aligned with DCASE Task 5 style.

Simplified implementation:
- Matches predicted and reference events using IoU and bipartite matching.
- Ignores time before the 5th POS endtime (caller must trim refs/preds accordingly).
- Treats UNK separately (caller should filter if needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Event:
    start: float
    end: float


def iou(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def match_events(refs: List[Event], preds: List[Event], iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Greedy matching by descending IoU (approximation of bipartite matching)."""
    if not refs or not preds:
        return 0, len(refs), len(preds)
    pairs = []
    for ri, r in enumerate(refs):
        for pi, p in enumerate(preds):
            pairs.append((iou(r, p), ri, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    matched_ref = set()
    matched_pred = set()
    tp = 0
    for score, ri, pi in pairs:
        if score < iou_thr:
            break
        if ri in matched_ref or pi in matched_pred:
            continue
        matched_ref.add(ri)
        matched_pred.add(pi)
        tp += 1
    fn = len(refs) - tp
    fp = len(preds) - tp
    return tp, fn, fp


def f_measure(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f, precision, recall

