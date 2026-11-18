"""CPU numerical references for single-token GQA decode with Split-KV."""

from math import sqrt


def dense_gqa(q, keys, values):
    import torch

    output = []
    for queries, key, value in zip(q, keys, values, strict=True):
        hkv, _, d = key.shape
        grouped = queries.float().reshape(hkv, -1, d)
        scores = torch.bmm(grouped, key.float().transpose(1, 2)) / sqrt(d)
        output.append(torch.bmm(scores.softmax(dim=-1), value.float()).flatten(0, 1))
    return torch.stack(output)


def split_gqa(q, keys, values, *, block_heads, splits, tile, probability_dtype=None):
    """Return FP32 output, split-normalized partial outputs and natural LSEs."""
    import torch

    if q.ndim != 3 or q.shape[0] != len(keys) or len(keys) != len(values):
        raise ValueError("expected batch-aligned Q, K and V")
    batch, hq, d = q.shape
    partial = torch.empty((batch, splits, hq, d), device=q.device)
    lse = torch.empty((batch, splits, hq), device=q.device)
    for b, (key, value) in enumerate(zip(keys, values, strict=True)):
        hkv, length, kd = key.shape
        if key.shape != value.shape or kd != d or hq % hkv or not 1 <= splits <= length:
            raise ValueError("invalid GQA dimensions or empty split")
        group = hq // hkv
        if not 1 <= block_heads <= group or tile <= 0:
            raise ValueError("invalid packed block or KV tile")
        for block in range(0, group, block_heads):
            limit = min(block_heads, group - block)
            qb = q[b].reshape(hkv, group, d)[:, block:block + limit].float()
            for s in range(splits):
                start, end = length * s // splits, length * (s + 1) // splits
                maximum = torch.full_like(qb[:, :, 0], -torch.inf)
                ell = torch.zeros_like(maximum)
                acc = torch.zeros_like(qb)
                for offset in range(start, end, tile):
                    stop = min(end, offset + tile)
                    k = key[:, offset:stop].float()
                    v = value[:, offset:stop].float()
                    scores = torch.bmm(qb, k.transpose(1, 2)) / sqrt(d)
                    new_max = torch.maximum(maximum, scores.amax(dim=-1))
                    alpha = torch.zeros_like(maximum) if offset == start else (maximum - new_max).exp()
                    p = (scores - new_max[..., None]).exp()
                    ell = alpha * ell + p.sum(dim=-1)
                    p_pv = p if probability_dtype is None else p.to(probability_dtype).float()
                    acc = alpha[..., None] * acc + torch.bmm(p_pv, v)
                    maximum = new_max
                partial[b, s].view(hkv, group, d)[:, block:block + limit] = acc / ell[..., None]
                lse[b, s].view(hkv, group)[:, block:block + limit] = maximum + ell.log()
    weights = (lse - torch.logsumexp(lse, dim=1, keepdim=True)).exp()
    return (weights[..., None] * partial).sum(dim=1), partial, lse
