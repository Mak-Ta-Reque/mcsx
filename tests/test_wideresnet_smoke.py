import os
import pytest

try:
    import torch
    from models.wideresnet import wideresnet28_10
except Exception:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch not installed in test environment")
@torch.no_grad()
def test_wrn28_10_forward_shape():
    model = wideresnet28_10(num_classes=10, depth=int(os.getenv('WRN_DEPTH','28')), widen_factor=int(os.getenv('WRN_WIDEN_FACTOR','10')))
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
