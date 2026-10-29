import pytest
import torch
from qd_attention.validation import validate_paged_metadata


def test_paged_metadata_rejects_invalid_values_and_collisions():
    pages = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    lengths = torch.tensor([5, 3], dtype=torch.int32)
    validate_paged_metadata(pages, lengths, block_size=4, num_blocks=3)
    with pytest.raises(ValueError, match='length'):
        validate_paged_metadata(pages, torch.tensor([9, 0], dtype=torch.int32), block_size=4, num_blocks=3)
    with pytest.raises(ValueError, match='block'):
        validate_paged_metadata(pages, lengths, block_size=4, num_blocks=2)
    with pytest.raises(ValueError, match='conflicting'):
        validate_paged_metadata(pages, lengths, block_size=4, num_blocks=3, positions=torch.tensor([0, 4], dtype=torch.int32))
