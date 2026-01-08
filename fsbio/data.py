"""dataset building and balancing."""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Tuple

import h5py
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def labels_to_int(labels: Iterable[str], class_set: Iterable[str]) -> np.ndarray:
    # map class names to indices
    label2idx = {label: index for index, label in enumerate(class_set)}
    return np.array([label2idx[label] for label in labels])


def balance_classes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # simple random oversampling
    x_index = [[index] for index in range(len(features))]
    ros = RandomOverSampler(random_state=42)
    x_unif, y_unif = ros.fit_resample(x_index, labels)
    indices = [idx[0] for idx in x_unif]
    return np.array([features[i] for i in indices]), np.array([labels[i] for i in indices])


def _norm_stats(features: np.ndarray) -> Tuple[float, float]:
    # compute global mean and std
    return float(np.mean(features)), float(np.std(features))


class DataBuilder:
    # handles train/val splits from hdf5

    def __init__(self, conf):
        hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
        hdf_train = h5py.File(hdf_path, 'r')
        self.x = hdf_train['features'][:]
        self.labels = [s.decode() for s in hdf_train['labels'][:]]

        class_set = set(self.labels)
        self.y = labels_to_int(self.labels, class_set)
        self.x, self.y = balance_classes(self.x, self.y)

        array_train = np.arange(len(self.x))
        _, _, _, _, train_array, valid_array = train_test_split(
            self.x, self.y, array_train, random_state=42, stratify=self.y
        )
        self.train_index = train_array
        self.valid_index = valid_array
        self.mean, self.std = _norm_stats(self.x[train_array])

    def _scale(self, feats: np.ndarray) -> np.ndarray:
        return (feats - self.mean) / self.std

    def generate_train(self):
        # returns normalized train and val splits
        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        x_train = self._scale(self.x[train_array])
        y_train = self.y[train_array]
        x_val = self._scale(self.x[valid_array])
        y_val = self.y[valid_array]
        return x_train, y_train, x_val, y_val


class EvalBuilder(DataBuilder):
    # wraps eval hdf5 features

    def __init__(self, hf, conf):
        super().__init__(conf=conf)
        self.x_pos = hf['feat_pos'][:]
        self.x_neg = hf['feat_neg'][:]
        self.x_query = hf['feat_query'][:]
        self.hop_seg = hf['hop_seg'][:]

    def generate_eval(self):
        # returns normalized eval features
        x_pos = self._scale(self.x_pos)
        x_neg = self._scale(self.x_neg)
        x_query = self._scale(self.x_query)
        return x_pos, x_neg, x_query, self.hop_seg
