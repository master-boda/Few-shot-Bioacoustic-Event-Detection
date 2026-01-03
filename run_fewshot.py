"""End-to-end few-shot detection respecting DCASE Task 5 rules."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torchaudio

from src.pipeline.fewshot import (
    load_audio,
    read_annotations,
    split_support_query,
    build_patches,
    compute_prototypes,
    score_queries,
    postprocess,
)
from src.utils.config import load_config
from src.models.protonet import ProtoNet


def process_file(
    audio_path: Path,
    ann_path: Path,
    cfg,
    encoder: ProtoNet,
    device: torch.device,
    verbose: bool = False,
) -> List[List]:
    sample_rate = int(cfg.get("sample_rate", 32000))
    feature_cfg = cfg.get("features", {})
    post_cfg = cfg.get("postprocess", {})

    wav = load_audio(audio_path, sample_rate)
    events = read_annotations(ann_path)
    support, ignore_before, _ = split_support_query(events, support_k=5)
    if verbose:
        print(f"[fewshot] {ann_path.name}: support classes={list(support.keys())}, ignore_before={ignore_before:.2f}")

    window_seconds = float(cfg.get("window_seconds", 1.0))
    hop_seconds = float(cfg.get("hop_seconds", 0.25))
    if cfg.get("adaptive_window", False):
        durations = [ev.end - ev.start for evs in support.values() for ev in evs]
        max_dur = max(durations, default=0.0)
        if max_dur > 0:
            window_seconds = max_dur
        min_sec = cfg.get("adaptive_min_seconds")
        max_sec = cfg.get("adaptive_max_seconds")
        if min_sec is not None:
            window_seconds = max(window_seconds, float(min_sec))
        if max_sec is not None:
            window_seconds = min(window_seconds, float(max_sec))
        ratio = cfg.get("adaptive_hop_ratio", 0.5)
        if ratio is not None:
            hop_seconds = max(window_seconds * float(ratio), 1e-4)

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 320)),
        n_mels=int(feature_cfg.get("n_mels", 64)),
        f_min=int(feature_cfg.get("f_min", 50)),
        f_max=int(feature_cfg.get("f_max", 14000)) if feature_cfg.get("f_max") is not None else None,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    support_patches, query_patches = build_patches(
        wav,
        sample_rate,
        support,
        ignore_before,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        melspec=melspec,
        to_db=to_db,
        use_spectral_contrast=bool(feature_cfg.get("use_spectral_contrast", False)),
        spectral_contrast_bands=int(feature_cfg.get("spectral_contrast_bands", 6)),
        normalize=str(feature_cfg.get("normalize", "none")),
    )

    prototypes = compute_prototypes(encoder, support_patches, device)
    scores = score_queries(
        encoder,
        prototypes,
        query_patches,
        device,
        metric="cosine",
    )

    threshold = float(post_cfg.get("threshold", 0.0))
    median_f = int(post_cfg.get("median_filter", 0))
    min_dur = float(post_cfg.get("min_duration", 0.0))

    rows: List[List] = []
    for cls, cls_scores in scores.items():
        events = postprocess(cls_scores, query_patches, threshold, median_f, min_dur, label=cls)
        for ev in events:
            rows.append([audio_path.name, ev.start, ev.end, cls, threshold])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Few-shot detection pipeline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--annotations_root", type=str, required=True, help="Root with per-file annotations (with label column)")
    parser.add_argument("--audio_root", type=str, required=True, help="Root where audio files reside")
    parser.add_argument("--output_csv", type=str, default="preds.csv")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    feature_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})
    encoder = ProtoNet(
        input_channels=2 if feature_cfg.get("use_spectral_contrast", False) else 1,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 1)),
        channel_mult=int(model_cfg.get("channel_mult", 2)),
        encoder_type=str(model_cfg.get("encoder", "simple_cnn")),
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(state["model"], strict=False)
    encoder.eval()

    ann_paths = sorted(Path(args.annotations_root).rglob("*.csv"))
    rows: List[List] = []
    for ann in ann_paths:
        rel = ann.relative_to(args.annotations_root)
        audio_path = Path(args.audio_root) / rel.with_suffix(".wav")
        file_rows = process_file(audio_path, ann, cfg, encoder, device, verbose=args.verbose)
        rows.extend(file_rows)
        if args.verbose:
            print(f"[fewshot] Processed {ann.name}, events: {len(file_rows)}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["Audiofilename", "Starttime", "Endtime", "Label", "Score"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} events to {out_path}")


if __name__ == "__main__":
    main()
