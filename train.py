"""Entry point for few-shot bioacoustic event detection experiments."""

from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import BioacousticEpisodeDataset
from src.models.protonet import ProtoNet
from src.training.loop import train_one_epoch, validate
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot bioacoustic event detection trainer")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_way = int(cfg["n_way"])
    k_shot = int(cfg["k_shot"])
    query_per_class = int(cfg["query_per_class"])
    episodes_per_epoch = int(cfg.get("episodes_per_epoch", 100))
    val_episodes = int(cfg.get("val_episodes", 50))
    feature_cfg = cfg.get("features", {})

    train_dataset = BioacousticEpisodeDataset(
        metadata_csv=cfg["train_metadata"],
        data_root=cfg["data_root"],
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=query_per_class,
        sample_rate=int(cfg.get("sample_rate", 32000)),
        clip_duration=float(cfg.get("clip_duration", 5.0)),
        n_mels=int(feature_cfg.get("n_mels", 64)),
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 320)),
        f_min=int(feature_cfg.get("f_min", 50)),
        f_max=int(feature_cfg.get("f_max", 14000)) if feature_cfg.get("f_max") is not None else None,
        episodes=episodes_per_epoch,
        use_spectral_contrast=bool(feature_cfg.get("use_spectral_contrast", False)),
        spectral_contrast_bands=int(feature_cfg.get("spectral_contrast_bands", 6)),
    )
    val_dataset = BioacousticEpisodeDataset(
        metadata_csv=cfg["val_metadata"],
        data_root=cfg["data_root"],
        n_way=n_way,
        k_shot=k_shot,
        query_per_class=query_per_class,
        sample_rate=int(cfg.get("sample_rate", 32000)),
        clip_duration=float(cfg.get("clip_duration", 5.0)),
        n_mels=int(feature_cfg.get("n_mels", 64)),
        n_fft=int(feature_cfg.get("n_fft", 1024)),
        hop_length=int(feature_cfg.get("hop_length", 320)),
        f_min=int(feature_cfg.get("f_min", 50)),
        f_max=int(feature_cfg.get("f_max", 14000)) if feature_cfg.get("f_max") is not None else None,
        episodes=val_episodes,
        use_spectral_contrast=bool(feature_cfg.get("use_spectral_contrast", False)),
        spectral_contrast_bands=int(feature_cfg.get("spectral_contrast_bands", 6)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model_cfg = cfg.get("model", {})
    model = ProtoNet(
        input_channels=getattr(train_dataset, "feature_channels", 1),
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 1)),
        channel_mult=int(model_cfg.get("channel_mult", 2)),
        encoder_type=str(model_cfg.get("encoder", "simple_cnn")),
    ).to(device)

    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(cfg.get("epochs", 1))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, n_way, device)
        metrics = validate(model, val_loader, n_way, device)
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_acc={metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
