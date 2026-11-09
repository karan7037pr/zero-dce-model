import torch

from zerodce.losses import ZeroDCELoss
from zerodce.model import DCEModel, enhance


def test_model_shapes_and_gradients():
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    model = DCEModel()
    enhanced, curve = model(image)
    assert enhanced.shape == image.shape
    assert curve.shape == (2, 24, 32, 32)
    losses = ZeroDCELoss()(enhanced, image, curve)
    losses["total"].backward()
    assert image.grad is not None


def test_curve_is_bounded():
    image = torch.rand(1, 3, 8, 8)
    curve = torch.randn(1, 24, 8, 8)
    output = enhance(image, curve)
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
