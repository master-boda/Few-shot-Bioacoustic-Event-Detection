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
import torchaudio
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


def filter_events(events, min_duration):
    if min_duration <= 0:
        return events
    return [ev for ev in events if (ev[1] - ev[0]) >= min_duration]


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


def load_wave_for_dataset(dataset, audio_path):
    wave, sr = dataset._load_wave(audio_path)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != dataset.sample_rate:
        wave = torchaudio.functional.resample(wave, sr, dataset.sample_rate)
    return wave


def sample_negative_segments(gaps, window_seconds, n_samples, rng):
    if not gaps or n_samples <= 0:
        return []
    spans = []
    for start, end in gaps:
        span = end - start - window_seconds
        if span > 0:
            spans.append((start, end, span))
    if not spans:
        return []
    total = sum(span for _, _, span in spans)
    if total <= 0:
        return []
    segments = []
    for _ in range(n_samples):
        r = rng.random() * total
        acc = 0.0
        for start, end, span in spans:
            acc += span
            if r <= acc:
                seg_start = start + rng.random() * span
                segments.append((seg_start, seg_start + window_seconds))
                break
    return segments


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
    parser.add_argument(
        "--score_mode",
        type=str,
        default=None,
        choices=["pos", "pos_minus_bg", "pos_vs_neg"],
        help="Score mode",
    )
    parser.add_argument("--bg_percentile", type=float, default=None, help="Bottom percentile for background prototype")
    parser.add_argument("--neg_samples", type=int, default=None, help="Negative samples per iteration")
    parser.add_argument("--neg_iterations", type=int, default=None, help="Negative prototype iterations")
    parser.add_argument("--threshold_mode", type=str, default=None, choices=["global", "percentile", "zscore"])
    parser.add_argument("--percentile", type=float, default=None, help="Percentile for per-file threshold")
    parser.add_argument("--zscore", type=float, default=None, help="Z-score multiplier for per-file threshold")
    parser.add_argument("--smooth", type=int, default=None, help="Median filter width (odd); 0 disables")
    parser.add_argument("--min_duration", type=float, default=None, help="Minimum event duration in seconds")
    parser.add_argument("--min_duration_mode", type=str, default=None, choices=["fixed", "adaptive"])
    parser.add_argument("--adaptive_min_ratio", type=float, default=None, help="Ratio of shortest shot for min duration")
    parser.add_argument("--event_mode", type=str, default=None, choices=["merge", "peak"], help="Event extraction mode")
    parser.add_argument("--peak_distance", type=float, default=None, help="Min distance between peaks (seconds)")
    parser.add_argument("--event_duration", type=float, default=None, help="Peak event duration (seconds)")
    parser.add_argument("--merge_gap", type=float, default=None, help="Merge gaps shorter than this (seconds)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    feature_cfg = cfg.get("features", {})
    perf_cfg = cfg.get("performance", {})
    if args.device.startswith("cuda") and bool(perf_cfg.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
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
        cache_wave=bool(perf_cfg.get("cache_wave", False)),
        normalize=str(feature_cfg.get("normalize", "none")),
        adaptive_window=bool(cfg.get("adaptive_window", False)),
        adaptive_hop_ratio=cfg.get("adaptive_hop_ratio", 0.5),
        adaptive_min_seconds=cfg.get("adaptive_min_seconds"),
        adaptive_max_seconds=cfg.get("adaptive_max_seconds"),
    )

    model_cfg = cfg.get("model", {})
    model = ProtoNet(
        input_channels=getattr(dataset, "feature_channels", 1),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 1)),
        channel_mult=int(model_cfg.get("channel_mult", 2)),
        encoder_type=str(model_cfg.get("encoder", "simple_cnn")),
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)
    model.eval()

    rows: List[List] = []
    use_amp = bool(perf_cfg.get("amp", False)) and args.device.startswith("cuda")
    with torch.no_grad():
        for item in dataset:
            support = item["support"].to(device)
            query = item["query"]
            query_meta = item["query_meta"]
            if query.numel() == 0:
                continue

            # Single-class prototype for class-of-interest vs background
            with torch.cuda.amp.autocast(enabled=use_amp):
                support_emb = model(support).cpu()
            prototype = support_emb.mean(dim=0, keepdim=True)

            q_emb_parts = []
            for i in range(0, query.shape[0], args.batch_size):
                q_chunk = query[i : i + args.batch_size].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    q_emb_parts.append(model(q_chunk).cpu())
            q_emb = torch.cat(q_emb_parts, dim=0)

            if args.metric == "cosine":
                q_norm = F.normalize(q_emb, dim=1)
                p_norm = F.normalize(prototype, dim=1)
                scores_pos = torch.matmul(q_norm, p_norm.t()).squeeze(1)
            else:
                scores_pos = -model.pairwise_distances(q_emb, prototype).squeeze(1)

            window_seconds = float(item.get("window_seconds", cfg.get("window_seconds", 1.0)))
            hop_seconds = float(item.get("hop_seconds", cfg.get("hop_seconds", 0.25)))
            score_mode = args.score_mode or post_cfg.get("score_mode", "pos")
            if score_mode == "pos_vs_neg":
                neg_samples = int(args.neg_samples if args.neg_samples is not None else post_cfg.get("neg_samples", 50))
                neg_iterations = int(args.neg_iterations if args.neg_iterations is not None else post_cfg.get("neg_iterations", 5))
                support_events = item.get("support_events", [])
                rng = np.random.default_rng(0)

                audio_path = query_meta[0].audio_path
                wave = load_wave_for_dataset(dataset, audio_path)
                support_sorted = sorted(support_events, key=lambda ev: ev.start)
                gaps = []
                if support_sorted:
                    if support_sorted[0].start > 0:
                        gaps.append((0.0, support_sorted[0].start))
                    for prev_ev, next_ev in zip(support_sorted[:-1], support_sorted[1:]):
                        if next_ev.start > prev_ev.end:
                            gaps.append((prev_ev.end, next_ev.start))
                neg_pool_size = max(neg_samples * max(1, neg_iterations), neg_samples)
                neg_segments = sample_negative_segments(gaps, window_seconds, neg_pool_size, rng)
                if not neg_segments:
                    scores = scores_pos
                else:
                    neg_feats = [dataset._extract_segment(wave, s, e) for s, e in neg_segments]
                    neg_feats = torch.stack(neg_feats).to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        neg_emb_all = model(neg_feats).cpu()
                    scores_iters = []
                    for _ in range(neg_iterations):
                        idx = rng.choice(len(neg_emb_all), size=neg_samples, replace=len(neg_emb_all) < neg_samples)
                        neg_proto = neg_emb_all[idx].mean(dim=0, keepdim=True)
                        if args.metric == "cosine":
                            n_norm = F.normalize(neg_proto, dim=1)
                            logits = torch.stack(
                                [
                                    scores_pos,
                                    torch.matmul(q_norm, n_norm.t()).squeeze(1),
                                ],
                                dim=1,
                            )
                        else:
                            d_neg = -model.pairwise_distances(q_emb, neg_proto).squeeze(1)
                            logits = torch.stack([scores_pos, d_neg], dim=1)
                        scores_iters.append(torch.softmax(logits, dim=1)[:, 0])
                    scores = torch.stack(scores_iters, dim=0).mean(dim=0)
            elif score_mode == "pos_minus_bg" and scores_pos.numel() > 1:
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
            min_mode = args.min_duration_mode or post_cfg.get("min_duration_mode", "fixed")
            if min_mode == "adaptive":
                support_events = item.get("support_events", [])
                if support_events:
                    min_support = min(ev.end - ev.start for ev in support_events)
                    ratio = args.adaptive_min_ratio if args.adaptive_min_ratio is not None else float(post_cfg.get("adaptive_min_ratio", 0.6))
                    min_duration = max(min_duration, ratio * min_support)
            event_mode = args.event_mode or post_cfg.get("event_mode", "merge")
            if event_mode == "peak":
                peak_distance = args.peak_distance if args.peak_distance is not None else float(post_cfg.get("peak_distance", 0.25))
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
                        event_duration = float(window_seconds)
                events = events_from_peaks(query_meta, scores, threshold=thr, event_duration=event_duration, min_distance_frames=min_distance_frames)
            else:
                events = merge_windows(query_meta, scores, threshold=thr, min_duration=min_duration)

            gap = args.merge_gap if args.merge_gap is not None else float(post_cfg.get("merge_gap", 0.0))
            events = merge_close_events(events, gap_seconds=gap)
            events = filter_events(events, min_duration)
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
