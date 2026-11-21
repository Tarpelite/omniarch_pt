"""Shape and causality checks for the paper model (small random configuration, no checkpoint needed)."""
import torch

from omniarch.paper import OmniArchPaper, PaperConfig


def small():
    torch.manual_seed(0)
    return OmniArchPaper(PaperConfig(hidden=64, layers=2, heads=4, intermediate=128, modes=4)).eval()


def test_shapes():
    m = small()
    for S in [(32,), (16, 16), (12, 12, 12)]:
        x = torch.randn(2, 3, 2, *S)
        axes = [torch.linspace(0, 1, n) for n in S]
        g = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)[None].expand(2, *S, len(S))
        with torch.no_grad():
            y = m(x, g)
        assert y.shape == x.shape


def test_block_causal():
    """Changing frame t must not change predictions made from frames < t."""
    m = small()
    S = (32,)
    g = torch.linspace(0, 1, 32)[None, :, None].expand(1, 32, 1)
    x = torch.randn(1, 4, 2, *S)
    x2 = x.clone()
    x2[:, 3] += 1.0
    with torch.no_grad():
        a, b = m(x, g), m(x2, g)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5)
    assert not torch.allclose(a[:, 3], b[:, 3])


def test_affine_equivariance():
    """Fields are normalised per frame, so an affine change of units carries through."""
    m = small()
    g = torch.linspace(0, 1, 32)[None, :, None].expand(1, 32, 1)
    x = torch.randn(1, 3, 1, 32)
    with torch.no_grad():
        a = m(x, g)
        b = m(3.0 * x + 7.0, g)
    assert torch.allclose(3.0 * a + 7.0, b, atol=1e-3)


def test_residual_affine_equivariance():
    """The residual decoder keeps the same unit equivariance."""
    torch.manual_seed(0)
    m = OmniArchPaper(PaperConfig(hidden=64, layers=2, heads=4, intermediate=128, modes=4, residual=True)).eval()
    g = torch.linspace(0, 1, 32)[None, :, None].expand(1, 32, 1)
    x = torch.randn(1, 3, 1, 32)
    with torch.no_grad():
        a = m(x, g)
        b = m(3.0 * x + 7.0, g)
    assert torch.allclose(3.0 * a + 7.0, b, atol=1e-3)
