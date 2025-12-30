"""Per-file few-shot training that mimics Task 5 inference.

For each annotated file, build support from the first 5 POS events and queries from
windows after the 5th POS end, label queries as POS/NEG via overlap, and optimize
a prototype-based detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.events import FewShotEventDataset, collect_annotations, Event
from src.models.protonet import ProtoNet
from src.utils.config import load_config


def label_queries(meta, pos_events: List[Event], iou_thr: float = 0.1) -> torch.Tensor:
    labels = []
    for info in meta:
        window = Event(start=info.start, end=info.end)
        labels.append(1 if max_iou(window, pos_events) >= iou_thr else 0)
    return torch.tensor(labels, dtype=torch.long)


def max_iou(win: Event, events: List[Event]) -> float:
    def iou(a: Event, b: Event) -> float:
        inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
        union = max(a.end, b.end) - min(a.start, b.start)
        return inter / union if union > 0 else 0.0
    return max((iou(win, e) for e in events), default=0.0)


def main():
    parser = argparse.ArgumentParser(description="Train proto detector with per-file 5-shot episodes")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--annotations_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_out", type=str, default=None, help="Path to save model weights")
    parser.add_argument("--checkpoint_in", type=str, default=None, help="Path to load model weights")
    parser.add_argument("--save_best", action="store_true", help="Save only when loss improves")
    parser.add_argument("--chunk_size", type=int, default=1, help="Chunk size for query windows to reduce memory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_cfg = cfg.get("features", {})
    ann_paths = collect_annotations(args.annotations_root)
    dataset = FewShotEventDataset(
        data_root=cfg["data_root"],
        annotation_paths=ann_paths,
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
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    model_cfg = cfg.get("model", {})
    model = ProtoNet(
        input_channels=getattr(dataset, "feature_channels", 1),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("optimizer", {}).get("lr", 1e-3)))
    if args.checkpoint_in:
        state = torch.load(args.checkpoint_in, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state.get("optimizer", optimizer.state_dict()))
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")

    if args.verbose:
        print(f"[train_events] files: {len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for idx, batch in enumerate(loader):
            support = batch["support"].squeeze(0).to(args.device)
            query = batch["query"].squeeze(0).to(args.device)
            meta = batch["query_meta"]  # list of WindowInfo
            if query.numel() == 0:
                continue

            pos_events = batch["pos_after_support"]
            if isinstance(pos_events, list) and len(pos_events) == 0:
                continue
            labels = label_queries(meta, pos_events if isinstance(pos_events, list) else [pos_events]).to(args.device)

            optimizer.zero_grad()
            num_chunks = max(1, (query.shape[0] + args.chunk_size - 1) // args.chunk_size)
            chunk_losses = 0.0
            for i in range(0, query.shape[0], args.chunk_size):
                q_chunk = query[i : i + args.chunk_size]
                lbl_chunk = labels[i : i + args.chunk_size]
                proto = model(support).mean(dim=0, keepdim=True)
                q_emb = model(q_chunk)
                dists = model.pairwise_distances(q_emb, proto).squeeze(1)
                # two-class logits: background=0, foreground=-dist
                logits = torch.stack([torch.zeros_like(dists), -dists], dim=1)
                chunk_loss = criterion(logits, lbl_chunk) / num_chunks
                chunk_loss.backward()
                chunk_losses += chunk_loss.item()
            optimizer.step()
            total_loss += chunk_losses
            if args.verbose:
                print(f"[train_events] epoch {epoch} batch {idx+1}/{len(loader)} loss={chunk_losses:.4f} file={batch.get('ann_path')}")
        print(f"Epoch {epoch}/{args.epochs} loss={total_loss/max(1,len(loader)):.4f}")
        if args.checkpoint_out:
            if args.save_best and total_loss >= best_loss:
                continue
            best_loss = min(best_loss, total_loss)
            out_path = Path(args.checkpoint_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, out_path)
            print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
