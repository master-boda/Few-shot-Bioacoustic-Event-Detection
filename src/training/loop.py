"""Simple training loop scaffold for few-shot episodes."""

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    n_way: int,
    device: torch.device,
) -> float:
    """Run one synthetic training epoch; replace with real loss computation."""
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        optimizer.zero_grad()
        support = batch["support"].squeeze(0).to(device)  # (N*K, 1, n_mels, time)
        query = batch["query"].squeeze(0).to(device)      # (N*Q, 1, n_mels, time)
        support_labels = batch["support_labels"].squeeze(0).to(device)
        query_labels = batch["query_labels"].squeeze(0).to(device)

        support_embeddings = model(support)
        query_embeddings = model(query)
        prototypes = model.compute_prototypes(support_embeddings, support_labels, n_way)
        dists = model.pairwise_distances(query_embeddings, prototypes)
        logits = -dists
        loss = criterion(logits, query_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def validate(
    model: nn.Module,
    loader: DataLoader,
    n_way: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate synthetic validation accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            query = batch["query"].squeeze(0).to(device)
            support = batch["support"].squeeze(0).to(device)
            support_labels = batch["support_labels"].squeeze(0).to(device)
            query_labels = batch["query_labels"].squeeze(0).to(device)
            prototypes = model.compute_prototypes(model(support), support_labels, n_way)
            logits = -model.pairwise_distances(model(query), prototypes)
            preds = logits.argmax(dim=1)
            correct += (preds == query_labels).sum().item()
            total += query_labels.numel()
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy}
