"""Few-shot event detection script aligned with DCASE Task 5 rules.

For each annotated file:
- Use the first 5 POS events as support to build prototypes.
- Slide windows over the rest of the file to classify query windows.
- Emit detected events (merged consecutive positive windows) to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import medfilt, find_peaks

from src.data.events import FewShotEventDataset, collect_annotations
from src.models.protonet import ProtoNet
from src.utils.config import load_config


def merge_windows(meta, scores, threshold, min_duration=0.0):
    """Merge consecutive windows with score >= threshold into events."""
    events = []
    current = None
    for info, score in zip(meta, scores):
        if score >= threshold:
            if current is None:
                current = [info.start, info.end]
            else:
                current[1] = info.end
        else:
            if current:
                if current[1] - current[0] >= min_duration:
                    events.append(current)
                current = None
    if current and current[1] - current[0] >= min_duration:
        events.append(current)
    return events


def merge_close_events(events, gap_seconds=0.0):
    if not events or gap_seconds <= 0:
        return events
    events = sorted(events, key=lambda x: x[0])
    merged = [events[0]]
    for start, end in events[1:]:
        if start - merged[-1][1] <= gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def events_from_peaks(meta, scores, threshold, event_duration, min_distance_frames):
    if len(scores) == 0:
        return []
    peaks, props = find_peaks(scores, height=threshold, distance=max(1, min_distance_frames))
    if len(peaks) == 0:
        return []
    start_limit = meta[0].start
    end_limit = meta[-1].end
    half = event_duration / 2.0
    events = []
    for idx in peaks:
        center = 0.5 * (meta[idx].start + meta[idx].end)
        start = max(start_limit, center - half)
        end = min(end_limit, center + half)
        if end > start:
            events.append([start, end])
    return events


def main():
    parser = argparse.ArgumentParser(description="Few-shot event detection (Task 5 style)")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--annotations_root", type=str, required=True, help="Root containing annotation CSVs (e.g., Validation_Set)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=None, help="Global threshold (used if threshold_mode=global)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=64, help="Chunk size for query windows during inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint from train_events.py")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"], help="Similarity metric")
    parser.add_argument("--score_mode", type=str, default=None, choices=["pos", "pos_minus_bg"], help="Score mode")
    parser.add_argument("--bg_percentile", type=float, default=None, help="Bottom percentile for background prototype")
    parser.add_argument("--threshold_mode", type=str, default=None, choices=["global", "percentile", "zscore"])
    parser.add_argument("--percentile", type=float, default=None, help="Percentile for per-file threshold")
    parser.add_argument("--zscore", type=float, default=None, help="Z-score multiplier for per-file threshold")
    parser.add_argument("--smooth", type=int, default=None, help="Median filter width (odd); 0 disables")
    parser.add_argument("--min_duration", type=float, default=None, help="Minimum event duration in seconds")
    parser.add_argument("--event_mode", type=str, default=None, choices=["merge", "peak"], help="Event extraction mode")
    parser.add_argument("--peak_distance", type=float, default=None, help="Min distance between peaks (seconds)")
    parser.add_argument("--event_duration", type=float, default=None, help="Peak event duration (seconds)")
    parser.add_argument("--merge_gap", type=float, default=None, help="Merge gaps shorter than this (seconds)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    feature_cfg = cfg.get("features", {})
    post_cfg = cfg.get("postprocess", {})
    dataset = FewShotEventDataset(
        data_root=cfg["data_root"],
        annotation_paths=collect_annotations(args.annotations_root),
        sample_rate=int(cfg.get("sample_rate", 32000)),
        n_mels=int(feature_cfg.get("n_mels", 64)),
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 320)),
        f_min=int(feature_cfg.get("f_min", 50)),
        f_max=int(feature_cfg.get("f_max", 14000)) if feature_cfg.get("f_max") is not None else None,
        support_per_file=5,
        window_seconds=float(cfg.get("window_seconds", 1.0)),
        hop_seconds=float(cfg.get("hop_seconds", 0.25)),
        use_spectral_contrast=bool(feature_cfg.get("use_spectral_contrast", False)),
        spectral_contrast_bands=int(feature_cfg.get("spectral_contrast_bands", 6)),
    )

    model_cfg = cfg.get("model", {})
    model = ProtoNet(
        input_channels=getattr(dataset, "feature_channels", 1),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 1)),
        channel_mult=int(model_cfg.get("channel_mult", 2)),
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()

    rows: List[List] = []
    with torch.no_grad():
        for item in dataset:
            support = item["support"].to(device)
            query = item["query"]
            query_meta = item["query_meta"]
            if query.numel() == 0:
                continue

            # Single-class prototype for class-of-interest vs background
            support_emb = model(support).cpu()
            prototype = support_emb.mean(dim=0, keepdim=True)

            q_emb_parts = []
            for i in range(0, query.shape[0], args.batch_size):
                q_chunk = query[i : i + args.batch_size].to(device)
                q_emb_parts.append(model(q_chunk).cpu())
            q_emb = torch.cat(q_emb_parts, dim=0)

            if args.metric == "cosine":
                q_norm = F.normalize(q_emb, dim=1)
                p_norm = F.normalize(prototype, dim=1)
                scores_pos = torch.matmul(q_norm, p_norm.t()).squeeze(1)
            else:
                scores_pos = -model.pairwise_distances(q_emb, prototype).squeeze(1)

            score_mode = args.score_mode or post_cfg.get("score_mode", "pos")
            if score_mode == "pos_minus_bg" and scores_pos.numel() > 1:
                bg_perc = args.bg_percentile if args.bg_percentile is not None else float(post_cfg.get("bg_percentile", 10.0))
                k = max(1, int(scores_pos.numel() * (bg_perc / 100.0)))
                idx = torch.topk(scores_pos, k, largest=False).indices
                bg_proto = q_emb[idx].mean(dim=0, keepdim=True)
                if args.metric == "cosine":
                    bg_norm = F.normalize(bg_proto, dim=1)
                    scores_bg = torch.matmul(q_norm, bg_norm.t()).squeeze(1)
                else:
                    scores_bg = -model.pairwise_distances(q_emb, bg_proto).squeeze(1)
                scores = scores_pos - scores_bg
            else:
                scores = scores_pos

            if args.verbose and scores.numel() > 0:
                audio_name = query_meta[0].audio_path.name
                print(
                    f"[detect] {audio_name}: score min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}"
                )

            scores = scores.numpy()
            # Optional smoothing
            smooth = args.smooth if args.smooth is not None else int(post_cfg.get("median_filter", 0))
            if smooth and smooth > 1:
                if smooth % 2 == 0:
                    smooth += 1
                scores = medfilt(scores, kernel_size=smooth)

            mode = args.threshold_mode or post_cfg.get("threshold_mode", "global")
            if mode == "percentile":
                perc = args.percentile if args.percentile is not None else float(post_cfg.get("percentile", 99.5))
                thr = float(np.percentile(scores, perc))
            elif mode == "zscore":
                z = args.zscore if args.zscore is not None else float(post_cfg.get("zscore", 2.5))
                mu = float(scores.mean())
                sigma = float(scores.std()) or 1e-6
                thr = mu + z * sigma
            else:
                thr = args.threshold if args.threshold is not None else float(post_cfg.get("threshold", 0.0))

            min_duration = args.min_duration if args.min_duration is not None else float(post_cfg.get("min_duration", 0.0))
            event_mode = args.event_mode or post_cfg.get("event_mode", "merge")
            if event_mode == "peak":
                peak_distance = args.peak_distance if args.peak_distance is not None else float(post_cfg.get("peak_distance", 0.25))
                hop_seconds = float(cfg.get("hop_seconds", 0.25))
                min_distance_frames = int(max(1.0, peak_distance / max(hop_seconds, 1e-6)))
                if args.event_duration is not None:
                    event_duration = float(args.event_duration)
                elif post_cfg.get("event_duration"):
                    event_duration = float(post_cfg.get("event_duration"))
                else:
                    support_events = item.get("support_events", [])
                    if support_events:
                        event_duration = float(np.mean([ev.end - ev.start for ev in support_events]))
                    else:
                        event_duration = float(cfg.get("window_seconds", 1.0))
                events = events_from_peaks(query_meta, scores, threshold=thr, event_duration=event_duration, min_distance_frames=min_distance_frames)
            else:
                events = merge_windows(query_meta, scores, threshold=thr, min_duration=min_duration)

            gap = args.merge_gap if args.merge_gap is not None else float(post_cfg.get("merge_gap", 0.0))
            events = merge_close_events(events, gap_seconds=gap)
            if args.verbose and len(scores) > 0:
                print(
                    f"[detect] threshold_mode={mode} threshold={thr:.3f} smooth={smooth} min_dur={min_duration} "
                    f"score_mode={score_mode} event_mode={event_mode}"
                )
            audio_name = query_meta[0].audio_path.name
            for start, end in events:
                rows.append([audio_name, start, end])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Audiofilename", "Starttime", "Endtime"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} detections to {out_path}")


if __name__ == "__main__":
    main()
