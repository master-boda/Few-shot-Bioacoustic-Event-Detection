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

import pandas as pd
import torch

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


def main():
    parser = argparse.ArgumentParser(description="Few-shot event detection (Task 5 style)")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--annotations_root", type=str, required=True, help="Root containing annotation CSVs (e.g., Validation_Set)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.0, help="Logit threshold for positive detection")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=64, help="Chunk size for query windows during inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint from train_events.py")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    feature_cfg = cfg.get("features", {})
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
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    rows: List[List] = []
    with torch.no_grad():
        for item in dataset:
            support = item["support"].to(device)
            query = item["query"].to(device)
            query_meta = item["query_meta"]
            if query.numel() == 0:
                continue

            # Single-class prototype for class-of-interest vs background
            support_emb = model(support)
            prototype = support_emb.mean(dim=0, keepdim=True)
            logits_parts = []
            for i in range(0, query.shape[0], args.batch_size):
                q_chunk = query[i : i + args.batch_size]
                q_emb = model(q_chunk)
                dists = model.pairwise_distances(q_emb, prototype)  # (chunk, 1)
                logits_parts.append(-dists.squeeze(1).cpu())
            logits = torch.cat(logits_parts, dim=0)

            events = merge_windows(query_meta, logits, threshold=args.threshold)
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
