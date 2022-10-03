import torch
import pytest
from src.models.basic_CNN import get_model


def test_sanity():
    assert 2 + 2 == 4


def test_dims_model():
    out = 4
    model = get_model(in_channels=3, output_dim=out)

    assert len(model(torch.empty(3, 3)) == out)
