import warnings
import torch
from .tome import (bipartite_soft_matching,
                   bipartite_hybrid_merge,
                   bipartite_hybrid_pure,
                   bipartite_hybrid_prune,
                   merge_wavg)

__all__ = [
    "evit",
    "pumer"
]


def evit(values: torch.Tensor,
          score: torch.Tensor,
          num_keep_node: int,
          preserve_length: bool,
          obj=None):
    B, N, D = values.shape
    if num_keep_node == N:
        return values
    # assert num_keep_node <= N, f"total seq_length {N} but want to keep {num_keep_node}"

    score_nocls = score.clone()
    if preserve_length:
        key_score = score_nocls.sort(dim=1, descending=True)[0][:, num_keep_node:num_keep_node + 1]
        score_mask = torch.where(score_nocls > key_score,
                                 torch.ones_like(score_nocls),
                                 torch.zeros_like(score_nocls))[..., None].expand_as(values)
        values *= score_mask
    else:
        sorted_score_key = score_nocls.sort(dim=1)[0]
        while True:
            key_score = sorted_score_key[:, num_keep_node:num_keep_node + 1]
            score_mask = torch.where(score_nocls > key_score,
                                     torch.ones_like(score_nocls),
                                     torch.zeros_like(score_nocls))[..., None].expand_as(values)
            try:
                values = values[score_mask == 0].reshape(B, -1, D)
            except RuntimeError as e:
                num_keep_node += 1
                warnings.warn("Adjust num_keep_node to {}".format(num_keep_node))
            else:
                break
    obj.score_mask = score_mask
    return values


def pumer(values, score, num_keep_node, preserve_length,
          return_mask=False,
          obj=None):
    B, N, D = values.shape
    merge, unmerge = bipartite_soft_matching(score, r=N - num_keep_node)
    merge_values, _ = merge_wavg(merge, values)
    dense_shape = merge_values.shape
    merge_values = unmerge(merge_values)
    obj.score_mask = torch.all(merge_values != 0, keepdim=True, dim=-1)
    if not preserve_length:
        merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values


def prune_only(values: torch.Tensor,
                 score: torch.Tensor,
                 num_keep_node: int,
                 preserve_length: bool,
                 obj=None):
    B, N, D = values.shape
    merge, unmerge = bipartite_hybrid_prune(score, r=N - num_keep_node)
    merge_values, _ = merge_wavg(merge, values)
    dense_shape = merge_values.shape
    merge_values = unmerge(merge_values)
    obj.score_mask = torch.all(merge_values != 0, keepdim=True, dim=-1)
    if not preserve_length:
        merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values


def merge_only(values, score, num_keep_node, preserve_length,
                 obj=None):
    B, N, D = values.shape
    merge, unmerge = bipartite_hybrid_merge(score, r=N - num_keep_node)
    merge_values, _ = merge_wavg(merge, values)
    dense_shape = merge_values.shape
    merge_values = unmerge(merge_values)
    obj.score_mask = torch.all(merge_values != 0, keepdim=True, dim=-1)
    if not preserve_length:
        merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values


def hybrid(values, score, num_keep_node, preserve_length,
                obj=None):
    B, N, D = values.shape
    merge, unmerge = bipartite_hybrid_pure(score, r=N - num_keep_node)
    merge_values, _ = merge_wavg(merge, values)
    dense_shape = merge_values.shape
    merge_values = unmerge(merge_values)
    obj.score_mask = torch.all(merge_values != 0, keepdim=True, dim=-1)
    if not preserve_length:
        merge_values = merge_values[merge_values.abs().sum(-1) != 0, :].reshape(dense_shape)

    return merge_values
