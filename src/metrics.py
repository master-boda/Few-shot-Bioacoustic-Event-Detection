"""prototype loss and evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import EvalBuilder
from .model import build_encoder


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # compute squared euclidean distance
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise ValueError("embedding dims do not match")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(embeddings: torch.Tensor, labels: torch.Tensor, n_support: int):
    # standard prototypical loss

    def support_idx(c):
        return labels_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    labels_cpu = labels.to('cpu')
    emb_cpu = embeddings.to('cpu')

    classes = torch.unique(labels_cpu)
    n_classes = len(classes)
    n_query = labels.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(support_idx, classes))
    prototypes = torch.stack([emb_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(
        list(map(lambda c: labels_cpu.eq(c).nonzero()[n_support:], classes))
    ).view(-1)
    query_samples = emb_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss, acc_val


def _probability(proto_pos: torch.Tensor, proto_neg: torch.Tensor, query_out: torch.Tensor):
    # convert distances to probability for positive class
    prototypes = torch.stack([proto_pos, proto_neg]).squeeze(1)
    dists = euclidean_dist(query_out, prototypes)
    logits = -dists
    prob = torch.softmax(logits, dim=1)
    return prob[:, 0].detach().cpu().tolist()


def _apply_postprocess(conf, avg_shot_len, onset, offset):
    if avg_shot_len is None:
        return onset, offset
    if not conf.eval.get("postprocess", True):
        return onset, offset
    min_frac = conf.eval.get("min_duration_frac", 0.0)
    merge_frac = conf.eval.get("merge_gap_frac", 0.0)
    if min_frac > 0.0:
        min_dur = avg_shot_len * float(min_frac)
        keep = (offset - onset) >= min_dur
        onset = onset[keep]
        offset = offset[keep]
    if merge_frac > 0.0 and len(onset) > 0:
        merge_gap = avg_shot_len * float(merge_frac)
        merged_onset = [onset[0]]
        merged_offset = [offset[0]]
        for start, end in zip(onset[1:], offset[1:]):
            if start - merged_offset[-1] <= merge_gap:
                merged_offset[-1] = max(merged_offset[-1], end)
            else:
                merged_onset.append(start)
                merged_offset.append(end)
        onset = np.array(merged_onset)
        offset = np.array(merged_offset)
    return onset, offset


def _onset_offset_from_prob(conf, prob_final, thresh, hop_seg, strt_index_query):
    prob_thresh = np.where(prob_final > thresh, 1, 0)
    changes = np.convolve(np.array([1, -1]), prob_thresh)
    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]
    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr
    onset = (onset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = (offset_frames) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query
    offset = offset + str_time_query
    return onset, offset


def _get_sweep_thresholds(conf):
    thresholds = conf.eval.get("sweep_thresholds")
    if thresholds is not None:
        return [float(t) for t in thresholds]
    start = float(conf.eval.get("sweep_start", 0.0))
    stop = float(conf.eval.get("sweep_stop", 1.0))
    step = float(conf.eval.get("sweep_step", 0.05))
    if step <= 0:
        raise ValueError("sweep_step must be > 0")
    return list(np.arange(start, stop + 1e-9, step))


def evaluate_prototypes(conf=None, hdf_eval=None, device=None, strt_index_query=None):
    # inference loop for a single file
    gen_eval = EvalBuilder(hdf_eval, conf)
    x_pos, x_neg, x_query, hop_seg = gen_eval.generate_eval()
    avg_shot_len = None
    if 'avg_shot_len' in hdf_eval:
        avg_shot_len = float(hdf_eval['avg_shot_len'][:][0])

    x_pos = torch.tensor(x_pos)
    y_pos = torch.zeros(x_pos.shape[0], dtype=torch.long)
    x_neg = torch.tensor(x_neg)
    y_neg = torch.zeros(x_neg.shape[0], dtype=torch.long)
    x_query = torch.tensor(x_query)
    y_query = torch.zeros(x_query.shape[0], dtype=torch.long)

    query_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_query, y_query),
        batch_size=conf.eval.query_batch_size,
        shuffle=False,
    )

    encoder = build_encoder(conf)

    if device == 'cpu':
        state_dict = torch.load(conf.path.best_model, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(conf.path.best_model)

    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    encoder.eval()

    # compute positive prototype
    pos_set_feat = []
    with torch.no_grad():
        for batch in tqdm(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_pos, y_pos))):
            x, _ = batch
            x = x.to(device)
            feat = encoder(x).cpu()
            pos_set_feat.append(feat.mean(dim=0))
    proto_pos = torch.stack(pos_set_feat, dim=0).mean(dim=0)

    transductive = bool(conf.eval.get("transductive", False))
    trans_steps = int(conf.eval.get("transductive_steps", 5))
    trans_temp = float(conf.eval.get("transductive_temp", 1.0))
    trans_weight = float(conf.eval.get("transductive_query_weight", 1.0))

    # cache query embeddings for transductive refinement and scoring
    query_embeddings = []
    with torch.no_grad():
        for batch in query_loader:
            x_q, _ = batch
            x_q = x_q.to(device)
            feat_q = encoder(x_q).cpu()
            query_embeddings.append(feat_q)
    query_embeddings = torch.cat(query_embeddings, dim=0)

    prob_comb = []
    for i in range(conf.eval.iterations):
        prob_pos_iter = []
        neg_indices = torch.randperm(len(x_neg))[:conf.eval.samples_neg]
        x_neg_iter = x_neg[neg_indices]
        neg_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(x_neg_iter, y_neg[: len(x_neg_iter)]),
            batch_size=conf.eval.negative_set_batch_size,
            shuffle=False,
        )
        neg_sum = torch.zeros_like(proto_pos).cpu()
        neg_count = 0
        with torch.no_grad():
            for batch in neg_loader:
                x_n, _ = batch
                x_n = x_n.to(device)
                feat_neg = encoder(x_n).cpu()
                neg_sum += feat_neg.sum(dim=0)
                neg_count += feat_neg.shape[0]
        proto_neg = (neg_sum / max(1, neg_count)).to(device)

        if transductive and query_embeddings.shape[0] > 0:
            proto_pos_t = proto_pos.clone()
            proto_neg_t = proto_neg.cpu().clone()
            for _ in range(max(trans_steps, 1)):
                prototypes = torch.stack([proto_pos_t, proto_neg_t]).to(query_embeddings.device)
                dists = euclidean_dist(query_embeddings, prototypes)
                logits = -dists / max(trans_temp, 1e-6)
                prob = torch.softmax(logits, dim=1)
                weights_pos = prob[:, 0]
                weights_neg = prob[:, 1]
                pos_weighted = (weights_pos.unsqueeze(1) * query_embeddings).sum(dim=0)
                neg_weighted = (weights_neg.unsqueeze(1) * query_embeddings).sum(dim=0)
                pos_denom = weights_pos.sum() + 1e-6
                neg_denom = weights_neg.sum() + 1e-6
                proto_pos_t = (proto_pos_t + trans_weight * (pos_weighted / pos_denom)) / (1.0 + trans_weight)
                proto_neg_t = (proto_neg_t + trans_weight * (neg_weighted / neg_denom)) / (1.0 + trans_weight)
            proto_pos = proto_pos_t
            proto_neg = proto_neg_t.to(device)

        with torch.no_grad():
            prob_pos_iter.extend(_probability(proto_pos, proto_neg.cpu(), query_embeddings))

        prob_comb.append(prob_pos_iter)
        print("Iteration number {}".format(i))

    prob_final = np.mean(np.array(prob_comb), axis=0)
    if conf.eval.get("sweep", False):
        onset_offset_map = {}
        for thresh in _get_sweep_thresholds(conf):
            onset, offset = _onset_offset_from_prob(conf, prob_final, thresh, hop_seg, strt_index_query)
            onset, offset = _apply_postprocess(conf, avg_shot_len, onset, offset)
            assert len(onset) == len(offset)
            onset_offset_map[thresh] = (onset, offset)
        return onset_offset_map

    thresh = conf.eval.threshold
    onset, offset = _onset_offset_from_prob(conf, prob_final, thresh, hop_seg, strt_index_query)
    onset, offset = _apply_postprocess(conf, avg_shot_len, onset, offset)
    assert len(onset) == len(offset)
    return onset, offset
