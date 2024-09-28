"""  ToMe   """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
        metric: torch.Tensor,
        r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    """
    protected = 0

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        """ >>> new merge policy  >>> """
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        """ <<<  new merge policy  <<< """

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # we sorted the index to preserve the order of sequence

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        unm_idx = unm_idx.sort(dim=1)[0]
        src_idx = src_idx.sort(dim=1)[0]
        # dst_idx = dst_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        if src.dim() == 3:
            n, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=-2)
        elif src.dim() == 4:
            """  ToMe with multi-head   """
            n, h, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)[:, None].repeat(1, h, 1, 1))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), src, reduce=mode)
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        if unm.dim() == 3:
            n, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        elif unm.dim() == 4:
            """  ToMe with multi-head   """
            n, h, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, h, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c)[:, None].repeat(1, h, 1, 1), src=unm)

        return out

    return merge, unmerge


def bipartite_hybrid_prune(
        metric: torch.Tensor,
        r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    """
    protected = 0

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        """ >>> new merge policy  >>> """
        metric = metric / metric.norm(dim=-1, keepdim=True)  # torch.Size([1, 1000, 2560])
        #####sort score by size, then give small score to a and large score to b
        sorted_metric, _ = metric.sort(dim=1, descending=True)
        mid_point = sorted_metric.size(1) // 2
        # Assign the smaller half to 'a' and the larger half to 'b'
        a, b = sorted_metric[:, mid_point:, :], sorted_metric[:, :mid_point, :]

        scores = a @ b.transpose(-1, -2)

        """ <<<  new merge policy  <<< """

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # we sorted the index to preserve the order of sequence

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        unm_idx = unm_idx.sort(dim=1)[0]
        src_idx = src_idx.sort(dim=1)[0]
        # dst_idx = dst_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        # print('src.dim()',src.dim())

        if src.dim() == 3:
            n, t1, c = src.shape

            # print('src before',src.shape)
            # print('dst before',dst.shape)

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            src_zero = torch.zeros_like(src)
            # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_zero, reduce=mode)

            # print('unm',unm.shape)
            # print('src',src.shape)
            # print('dst',dst.shape)

            cat_out = torch.cat([unm, dst], dim=-2)
            # print('cat_out', cat_out.shape)

            ## Set the values at src_idx to zero
            # dst = dst.scatter_(-2, dst_idx.expand(n, r, c), torch.zeros_like(src))

            return cat_out  # torch.cat([unm, dst], dim=-2)
        elif src.dim() == 4:
            """  ToMe with multi-head   """
            n, h, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)[:, None].repeat(1, h, 1, 1))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1))
            src_zero = torch.zero_like(src)
            # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), src, reduce=mode)
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), src_zero, reduce=mode)

            # print('unm',unm.shape)
            # print('src',src.shape)
            # print('dst',dst.shape)

            cat_out = torch.cat([unm, dst], dim=-2)
            # print('cat_out', cat_out.shape)

            # Set the values at src_idx to zero
            # dst = dst.scatter_(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), torch.zeros_like(src))

            return cat_out  # torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        if unm.dim() == 3:
            n, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        elif unm.dim() == 4:
            """  ToMe with multi-head   """
            n, h, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, h, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c)[:, None].repeat(1, h, 1, 1), src=unm)

        return out

    return merge, unmerge


def bipartite_hybrid_merge(
        metric: torch.Tensor,
        r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    """
    protected = 0

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        """ >>> new merge policy  >>> """
        metric = metric / metric.norm(dim=-1, keepdim=True)  # torch.Size([1, 1000, 2560])
        #####sort score by size, then give small score to a and large score to b
        sorted_metric, _ = metric.sort(dim=1, descending=True)
        mid_point = sorted_metric.size(1) // 2
        # Assign the smaller half to 'a' and the larger half to 'b'
        a, b = sorted_metric[:, mid_point:, :], sorted_metric[:, :mid_point, :]

        scores = a @ b.transpose(-1, -2)

        """ <<<  new merge policy  <<< """

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # we sorted the index to preserve the order of sequence

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        unm_idx = unm_idx.sort(dim=1)[0]
        src_idx = src_idx.sort(dim=1)[0]
        # dst_idx = dst_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        if src.dim() == 3:
            n, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=-2)
        elif src.dim() == 4:
            """  ToMe with multi-head   """
            n, h, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)[:, None].repeat(1, h, 1, 1))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), src, reduce=mode)
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        if unm.dim() == 3:
            n, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        elif unm.dim() == 4:
            """  ToMe with multi-head   """
            n, h, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, h, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c)[:, None].repeat(1, h, 1, 1), src=unm)

        return out

    return merge, unmerge


def bipartite_hybrid_pure(
        metric: torch.Tensor,
        r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    """
    protected = 0

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        """ >>> new merge policy  >>> """
        metric = metric / metric.norm(dim=-1, keepdim=True)  # torch.Size([1, 1000, 2560])
        #####sort score by size, then give small score to a and large score to b
        sorted_metric, _ = metric.sort(dim=1, descending=True)
        mid_point = sorted_metric.size(1) // 2
        # Assign the smaller half to 'a' and the larger half to 'b'
        a, b = sorted_metric[:, mid_point:, :], sorted_metric[:, :mid_point, :]

        scores = a @ b.transpose(-1, -2)

        """ <<<  new merge policy  <<< """

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # we sorted the index to preserve the order of sequence

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        unm_idx = unm_idx.sort(dim=1)[0]
        src_idx = src_idx.sort(dim=1)[0]
        # dst_idx = dst_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        if src.dim() == 3:
            n, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

            k = r // 2  # Example: Modify top half of r
            _, topk_indices = src.topk(k, dim=1)
            # src_zero = torch.zeros_like(src)
            # src.scatter_(-2, topk_indices, src_zero)
            mask = torch.zeros_like(src).scatter_(1, topk_indices, 1)
            src = src * mask

            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=-2)
        elif src.dim() == 4:
            """  ToMe with multi-head   """
            n, h, t1, c = src.shape

            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c)[:, None].repeat(1, h, 1, 1))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1))

            k = r // 2  # Example: Modify top half of r
            _, topk_indices = src.topk(k, dim=1)
            # src_zero = torch.zeros_like(src)
            # src.scatter_(-2, topk_indices, src_zero)
            mask = torch.zeros_like(src).scatter_(1, topk_indices, 1)
            src = src * mask

            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c)[:, None].repeat(1, h, 1, 1), src, reduce=mode)
            return torch.cat([unm, dst], dim=-2)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        if unm.dim() == 3:
            n, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        elif unm.dim() == 4:
            """  ToMe with multi-head   """
            n, h, _, c = unm.shape

            # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

            out = torch.zeros(n, h, metric.shape[1], c, device=x.device, dtype=x.dtype)

            out[..., 1::2, :] = dst

            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c)[:, None].repeat(1, h, 1, 1), src=unm)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
        metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
        metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
        merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
        merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
