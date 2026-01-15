"""episodic sampler for few-shot training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class EpisodicBatchSampler(Sampler):
    # sample n-way episodes with equal shots per class

    def __init__(self, labels, n_episodes, n_way, n_samples):
        super().__init__(None)
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_samples = n_samples
        self.labels = np.array(labels)

        self.classes = np.unique(self.labels)
        self.indexes = {}
        for cls in self.classes:
            self.indexes[cls] = np.where(self.labels == cls)[0]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            batch = []
            for cls in selected_classes:
                idx = np.random.choice(self.indexes[cls], self.n_samples, replace=True)
                batch.extend(idx)
            yield torch.tensor(batch)
