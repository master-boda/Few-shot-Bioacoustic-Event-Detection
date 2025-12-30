"""Closer-to-official event-based evaluation helpers for DCASE Task 5.

Implements IoU-based bipartite matching via Hungarian algorithm and supports
ignoring the region before the 5th POS endtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Event:
    start: float
    end: float


def load_events(path: Path, drop_unk: bool = True) -> List[Event]:
    """Load events from CSV. If 'Q' or a label column exists, drop UNK rows when drop_unk is True."""
    df = pd.read_csv(path)
    start_col = "Starttime" if "Starttime" in df.columns else "Start"
    end_col = "Endtime" if "Endtime" in df.columns else "End"
    label_cols = [c for c in df.columns if c not in {start_col, end_col, "Audiofilename"}]
    if drop_unk and label_cols:
        # Keep rows where any label col is POS
        mask = False
        for c in label_cols:
            mask |= df[c] == "POS"
        df = df[mask]
    return [Event(float(s), float(e)) for s, e in zip(df[start_col], df[end_col])]


def fifth_pos_end(path: Path) -> float:
    """Return the end time of the 5th POS event (or 0.0 if fewer)."""
    df = pd.read_csv(path)
    start_col = "Starttime" if "Starttime" in df.columns else "Start"
    end_col = "Endtime" if "Endtime" in df.columns else "End"
    label_cols = [c for c in df.columns if c not in {start_col, end_col, "Audiofilename"}]
    if label_cols:
        mask = False
        for c in label_cols:
            mask |= df[c] == "POS"
        df = df[mask]
    if len(df) < 5:
        return 0.0
    return float(df.iloc[4][end_col])


def iou_matrix(refs: List[Event], preds: List[Event]) -> np.ndarray:
    mat = np.zeros((len(refs), len(preds)), dtype=np.float32)
    for i, r in enumerate(refs):
        for j, p in enumerate(preds):
            inter = max(0.0, min(r.end, p.end) - max(r.start, p.start))
            union = max(r.end, p.end) - min(r.start, p.start)
            mat[i, j] = inter / union if union > 0 else 0.0
    return mat


def match_events_hungarian(refs: List[Event], preds: List[Event], iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Maximum bipartite matching using Hungarian on IoU matrix."""
    if not refs or not preds:
        return 0, len(refs), len(preds)
    mat = iou_matrix(refs, preds)
    cost = 1.0 - mat  # maximize IoU == minimize (1 - IoU)
    row_ind, col_ind = linear_sum_assignment(cost)
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if mat[r, c] >= iou_thr:
            tp += 1
    fn = len(refs) - tp
    fp = len(preds) - tp
    return tp, fn, fp


def f_measure(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f, precision, recall

