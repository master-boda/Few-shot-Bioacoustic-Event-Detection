"""Episode-style proto training on POS segments with random negatives.

For each annotated file, build support from POS events, extract additional POS
patches from the remaining events, and sample NEG patches from non-event
regions of the full file. Optimize a prototypical classifier (no BCE head).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.data.events import FewShotEpisodeDataset, collect_annotations
from src.models.protonet import ProtoNet
from src.utils.config import load_config


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
    parser.add_argument("--support_per_file", type=int, default=None, help="Support shots per file")
    parser.add_argument("--negatives_per_file", type=int, default=None, help="Negative patches per file")
    parser.add_argument("--max_pos_patches", type=int, default=None, help="Max positive patches per file")
    parser.add_argument("--support_strategy", type=str, default=None, choices=["first", "random"])
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_cfg = cfg.get("features", {})
    train_cfg = cfg.get("train_events", {})
    perf_cfg = cfg.get("performance", {})
    ann_paths = collect_annotations(args.annotations_root)
    support_per_file = int(args.support_per_file if args.support_per_file is not None else train_cfg.get("support_per_file", 5))
    negatives_per_file = int(args.negatives_per_file if args.negatives_per_file is not None else train_cfg.get("negatives_per_file", 50))
    max_pos_patches = args.max_pos_patches if args.max_pos_patches is not None else train_cfg.get("max_pos_patches")
    support_strategy = str(args.support_strategy if args.support_strategy is not None else train_cfg.get("support_strategy", "first"))

    dataset = FewShotEpisodeDataset(
        data_root=cfg["data_root"],
        annotation_paths=ann_paths,
        sample_rate=int(cfg.get("sample_rate", 32000)),
        n_mels=int(feature_cfg.get("n_mels", 64)),
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 320)),
        f_min=int(feature_cfg.get("f_min", 50)),
        f_max=int(feature_cfg.get("f_max", 14000)) if feature_cfg.get("f_max") is not None else None,
        support_per_file=support_per_file,
        window_seconds=float(cfg.get("window_seconds", 1.0)),
        hop_seconds=float(cfg.get("hop_seconds", 0.25)),
        use_spectral_contrast=bool(feature_cfg.get("use_spectral_contrast", False)),
        spectral_contrast_bands=int(feature_cfg.get("spectral_contrast_bands", 6)),
        class_strategy=str(train_cfg.get("class_strategy", "max_pos")),
        cache_wave=bool(perf_cfg.get("cache_wave", False)),
        normalize=str(feature_cfg.get("normalize", "none")),
        negatives_per_file=negatives_per_file,
        max_pos_patches=max_pos_patches,
        support_strategy=support_strategy,
    )
    num_workers = int(perf_cfg.get("num_workers", 0))
    pin_memory = bool(perf_cfg.get("pin_memory", False))
    persistent_workers = bool(perf_cfg.get("persistent_workers", True)) if num_workers > 0 else False
    prefetch_factor = int(perf_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model_cfg = cfg.get("model", {})
    model = ProtoNet(
        input_channels=getattr(dataset, "feature_channels", 1),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 1)),
        channel_mult=int(model_cfg.get("channel_mult", 2)),
        encoder_type=str(model_cfg.get("encoder", "simple_cnn")),
    ).to(args.device)
    if args.device.startswith("cuda") and bool(perf_cfg.get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("optimizer", {}).get("lr", 1e-3)))
    if args.checkpoint_in:
        state = torch.load(args.checkpoint_in, map_location=args.device)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state.get("optimizer", optimizer.state_dict()))
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float("inf")

    use_amp = bool(perf_cfg.get("amp", False)) and args.device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    if args.verbose:
        print(
            "[train_events] files={} support_per_file={} negs_per_file={} max_pos_patches={}".format(
                len(dataset),
                support_per_file,
                negatives_per_file,
                max_pos_patches,
            )
        )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for idx, batch in enumerate(loader):
            support = batch["support"].to(args.device)
            query_pos = batch["query_pos"].to(args.device)
            query_neg = batch["query_neg"].to(args.device)
            if query_pos.numel() == 0 or query_neg.numel() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                support_emb = model(support)
                proto_pos = support_emb.mean(dim=0, keepdim=True)

                pos_chunks = []
                for i in range(0, query_pos.shape[0], args.chunk_size):
                    pos_chunks.append(model(query_pos[i : i + args.chunk_size]))
                pos_emb = torch.cat(pos_chunks, dim=0) if pos_chunks else torch.empty(0, support_emb.shape[-1])

                neg_chunks = []
                for i in range(0, query_neg.shape[0], args.chunk_size):
                    neg_chunks.append(model(query_neg[i : i + args.chunk_size]))
                neg_emb = torch.cat(neg_chunks, dim=0) if neg_chunks else torch.empty(0, support_emb.shape[-1])

                if pos_emb.numel() == 0 or neg_emb.numel() == 0:
                    continue

                proto_neg = neg_emb.mean(dim=0, keepdim=True)

                chunk_losses = 0.0
                chunk_count = 0
                for emb, label_value in ((pos_emb, 1), (neg_emb, 0)):
                    for i in range(0, emb.shape[0], args.chunk_size):
                        e_chunk = emb[i : i + args.chunk_size]
                        d_neg = model.pairwise_distances(e_chunk, proto_neg).squeeze(1)
                        d_pos = model.pairwise_distances(e_chunk, proto_pos).squeeze(1)
                        logits = torch.stack([-d_neg, -d_pos], dim=1)
                        labels = torch.full((e_chunk.shape[0],), label_value, dtype=torch.long, device=args.device)
                        chunk_losses += criterion(logits, labels)
                        chunk_count += 1
                chunk_losses = chunk_losses / max(1, chunk_count)
            scaler.scale(chunk_losses).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(chunk_losses.item())
            if args.verbose:
                print(
                    "[train_events] epoch {}/{} batch {}/{} loss={:.4f} pos={} neg={} file={}".format(
                        epoch,
                        args.epochs,
                        idx + 1,
                        len(loader),
                        float(chunk_losses.item()),
                        query_pos.shape[0],
                        query_neg.shape[0],
                        batch.get("ann_path"),
                    )
                )
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
