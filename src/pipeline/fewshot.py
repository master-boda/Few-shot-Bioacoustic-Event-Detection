"""Lightweight few-shot bioacoustic event detection pipeline (Task 5-style).

Steps per file:
- Load audio, normalize, resample.
- Build support set: first K POS events per class.
- Build query set: sliding windows after the 5th POS end time (across classes).
- Encode support/query with a shared encoder; compute class prototypes.
- Score queries by distance to prototypes; post-process scores into events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import torch.nn.functional as F
from scipy.signal import medfilt
from torch import Tensor

from src.models.protonet import ProtoNet


@dataclass
class Event:
    start: float
    end: float
    label: str


@dataclass
class Patch:
    start: float
    end: float
    features: Tensor  # (channels, n_mels, time)


def load_audio(path: Path, sample_rate: int) -> Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    # peak normalize
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak
    return wav


def read_annotations(path: Path) -> List[Event]:
    df = pd.read_csv(path)
    start_col = "Starttime" if "Starttime" in df.columns else "Start"
    end_col = "Endtime" if "Endtime" in df.columns else "End"
    label_col = "label" if "label" in df.columns else (df.columns[-1] if df.columns[-1] not in {start_col, end_col, "Audiofilename"} else None)
    if label_col is None:
        raise ValueError(f"No label column found in {path}")
    events = []
    for _, row in df.iterrows():
        label = str(row[label_col])
        if label.upper() == "UNK":
            continue
        events.append(Event(float(row[start_col]), float(row[end_col]), label))
    return sorted(events, key=lambda e: e.start)


def split_support_query(events: List[Event], support_k: int = 5) -> Tuple[Dict[str, List[Event]], float, List[Event]]:
    """Return support events per class, ignore_before time (5th POS end across classes), and remaining events."""
    support: Dict[str, List[Event]] = {}
    for ev in events:
        support.setdefault(ev.label, [])
        if len(support[ev.label]) < support_k:
            support[ev.label].append(ev)
    fifth_ends = [evs[support_k - 1].end for evs in support.values() if len(evs) >= support_k]
    ignore_before = max(fifth_ends) if fifth_ends else 0.0
    remaining = [ev for ev in events if ev.end > ignore_before]
    return support, ignore_before, remaining


def extract_features(
    wav: Tensor,
    sample_rate: int,
    start_s: float,
    end_s: float,
    window_samples: int,
    melspec,
    to_db,
    use_spectral_contrast: bool,
    spectral_contrast_bands: int,
    normalize: str,
) -> Tensor:
    start = int(start_s * sample_rate)
    end = int(end_s * sample_rate)
    segment = wav[..., start:end]
    if segment.shape[-1] < window_samples:
        pad = window_samples - segment.shape[-1]
        segment = F.pad(segment, (0, pad))
    elif segment.shape[-1] > window_samples:
        segment = segment[..., :window_samples]
    mel = melspec(segment)
    log_mel = to_db(mel).squeeze(0)  # (n_mels, time)
    log_mel = log_mel.unsqueeze(0)  # channel dim
    if not use_spectral_contrast:
        return _normalize_features(log_mel, normalize)
    np_wave = segment.squeeze(0).numpy()
    sc = librosa.feature.spectral_contrast(
        y=np_wave,
        sr=sample_rate,
        n_fft=melspec.n_fft,
        hop_length=melspec.hop_length,
        n_bands=spectral_contrast_bands,
    )
    sc_t = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # (1,1,bands+1,frames)
    sc_resized = F.interpolate(sc_t, size=log_mel.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    feats = torch.cat([log_mel, sc_resized], dim=0)
    return _normalize_features(feats, normalize)


def _normalize_features(feats: Tensor, normalize: str) -> Tensor:
    if normalize != "minmax":
        return feats
    feat_min = feats.amin(dim=(1, 2), keepdim=True)
    feat_max = feats.amax(dim=(1, 2), keepdim=True)
    return (feats - feat_min) / (feat_max - feat_min + 1e-6)


def build_patches(
    wav: Tensor,
    sample_rate: int,
    support: Dict[str, List[Event]],
    ignore_before: float,
    window_seconds: float,
    hop_seconds: float,
    melspec,
    to_db,
    use_spectral_contrast: bool,
    spectral_contrast_bands: int,
    normalize: str,
) -> Tuple[Dict[str, List[Patch]], List[Patch]]:
    window_samples = int(window_seconds * sample_rate)
    support_patches: Dict[str, List[Patch]] = {}
    for cls, evs in support.items():
        support_patches[cls] = []
        for ev in evs:
            feat = extract_features(
                wav,
                sample_rate,
                ev.start,
                ev.end,
                window_samples,
                melspec,
                to_db,
                use_spectral_contrast,
                spectral_contrast_bands,
                normalize,
            )
            support_patches[cls].append(Patch(start=ev.start, end=ev.end, features=feat))

    query_patches: List[Patch] = []
    total_len = wav.shape[-1] / sample_rate
    t = ignore_before
    while t < total_len:
        start = t
        end = t + window_seconds
        feat = extract_features(
            wav,
            sample_rate,
            start,
            end,
            window_samples,
            melspec,
            to_db,
            use_spectral_contrast,
            spectral_contrast_bands,
            normalize,
        )
        query_patches.append(Patch(start=start, end=end, features=feat))
        t += hop_seconds
    return support_patches, query_patches


def compute_prototypes(encoder: ProtoNet, support: Dict[str, List[Patch]], device: torch.device) -> Dict[str, Tensor]:
    prototypes: Dict[str, Tensor] = {}
    for cls, patches in support.items():
        if not patches:
            continue
        feats = torch.stack([p.features for p in patches]).to(device)
        emb = encoder(feats)
        prototypes[cls] = emb.mean(dim=0)
    return prototypes


def score_queries(
    encoder: ProtoNet,
    prototypes: Dict[str, Tensor],
    queries: List[Patch],
    device: torch.device,
    metric: str = "cosine",
) -> Dict[str, np.ndarray]:
    if not prototypes or not queries:
        return {}
    q_feats = torch.stack([q.features for q in queries]).to(device)
    with torch.no_grad():
        q_emb = encoder(q_feats)  # (Q, D)
        scores: Dict[str, np.ndarray] = {}
        for cls, proto in prototypes.items():
            p = proto.to(device)
            if metric == "cosine":
                qe = F.normalize(q_emb, dim=1)
                pe = F.normalize(p, dim=0)
                s = torch.matmul(qe, pe)
                scores[cls] = s.detach().cpu().numpy()
            else:
                d = torch.cdist(q_emb, p.unsqueeze(0)).squeeze(1)
                scores[cls] = (-d).detach().cpu().numpy()
    return scores


def postprocess(
    scores: np.ndarray,
    patches: List[Patch],
    threshold: float,
    median_filter: int = 0,
    min_duration: float = 0.0,
    label: str = "",
) -> List[Event]:
    if median_filter and median_filter > 1:
        scores = medfilt(scores, kernel_size=median_filter)
    active = scores >= threshold
    events: List[Event] = []
    if not len(patches):
        return events
    start = None
    end = None
    for flag, patch in zip(active, patches):
        if flag and start is None:
            start = patch.start
            end = patch.end
        elif flag and start is not None:
            end = patch.end
        elif (not flag) and start is not None:
            if end - start >= min_duration:
                events.append(Event(start, end, label=label))
            start = None
            end = None
    if start is not None and end - start >= min_duration:
        events.append(Event(start, end, label=label))
    return events
