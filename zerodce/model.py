"""Zero-DCE enhancement network and differentiable enhancement curve."""

from __future__ import annotations

import torch
from torch import nn


def enhance(image: torch.Tensor, curve: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """Apply the Zero-DCE pixel-wise curve repeatedly.

    Args:
        image: Tensor shaped ``[N, 3, H, W]`` with values in ``[0, 1]``.
        curve: Tensor shaped ``[N, 24, H, W]`` (eight RGB curves).
        iterations: Number of curve applications to perform.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [N, 3, H, W]")
    if curve.ndim != 4 or curve.shape[1] != 24:
        raise ValueError("curve must have shape [N, 24, H, W]")
    result = image
    for i in range(iterations):
        alpha = curve[:, i * 3 : (i + 1) * 3]
        result = result + alpha * result * (1.0 - result)
    return result.clamp(0.0, 1.0)


class DCEModel(nn.Module):
    """The seven-convolution DCE-Net from Zero-DCE."""

    def __init__(self, curve_iterations: int = 8) -> None:
        super().__init__()
        self.curve_iterations = curve_iterations
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 24, 3, padding=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.conv1(image))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curve = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        enhanced = enhance(image, curve, self.curve_iterations)
        return enhanced, curve
