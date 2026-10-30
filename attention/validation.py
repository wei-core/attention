"""Opt-in, synchronizing metadata validation for debugging CUDA inputs."""
import torch


def validate_paged_metadata(block_table, context_lens, *, block_size, num_blocks, positions=None):
    """Validate page bounds and lengths. If appending, also reject duplicate destinations.

    This intentionally synchronizes CUDA metadata and is unsuitable for a hot path.
    The caller still owns inter-request page lifetime and cross-stream synchronization.
    """
    if block_size <= 0 or num_blocks <= 0:
        raise ValueError('block_size and num_blocks must be positive')
    if block_table.ndim != 2 or context_lens.ndim != 1 or block_table.shape[0] != context_lens.numel():
        raise ValueError('metadata shapes must be [B,MaxPages] and [B]')
    if block_table.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise ValueError('metadata must be int32')
    if positions is not None and (positions.shape != context_lens.shape or positions.dtype != torch.int32):
        raise ValueError('positions must be int32 [B]')
    pages = block_table.detach().cpu()
    lengths = context_lens.detach().cpu().tolist()
    pos = positions.detach().cpu().tolist() if positions is not None else None
    destinations = set()
    for b, length in enumerate(lengths):
        if not 0 <= length <= pages.shape[1] * block_size:
            raise ValueError(f'invalid length at batch {b}')
        for p in range((length + block_size - 1) // block_size):
            if not 0 <= int(pages[b, p]) < num_blocks:
                raise ValueError(f'invalid block at batch {b}, page {p}')
        if pos is not None:
            t = pos[b]
            if not 0 <= t < pages.shape[1] * block_size:
                raise ValueError(f'invalid append position at batch {b}')
            page = int(pages[b, t // block_size])
            if not 0 <= page < num_blocks:
                raise ValueError(f'invalid append page at batch {b}')
            destination = (page, t % block_size)
            if destination in destinations:
                raise ValueError('conflicting append destinations')
            destinations.add(destination)
