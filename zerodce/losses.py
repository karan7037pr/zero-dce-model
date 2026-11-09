"""The four zero-reference losses used to train Zero-DCE."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _mean_abs(x: torch.Tensor) -> torch.Tensor:
    return x.abs().mean()


class SpatialConsistencyLoss(nn.Module):
    def __init__(self, pool_size: int = 4) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        enhanced_gray = enhanced.mean(dim=1, keepdim=True)
        original_gray = original.mean(dim=1, keepdim=True)
        enhanced_gray = self.pool(enhanced_gray)
        original_gray = self.pool(original_gray)
        dh_e = enhanced_gray[:, :, :, 1:] - enhanced_gray[:, :, :, :-1]
        dh_o = original_gray[:, :, :, 1:] - original_gray[:, :, :, :-1]
        dv_e = enhanced_gray[:, :, 1:, :] - enhanced_gray[:, :, :-1, :]
        dv_o = original_gray[:, :, 1:, :] - original_gray[:, :, :-1, :]
        return _mean_abs(dh_e - dh_o) + _mean_abs(dv_e - dv_o)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size: int = 16, target: float = 0.6) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)
        self.target = target

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        mean = self.pool(enhanced.mean(dim=1, keepdim=True))
        return (mean - self.target).abs().mean()


class ColorConstancyLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        means = enhanced.mean(dim=(2, 3))
        rg = means[:, 0] - means[:, 1]
        rb = means[:, 0] - means[:, 2]
        gb = means[:, 1] - means[:, 2]
        return torch.sqrt(rg.square() + rb.square() + gb.square() + 1e-8).mean()


class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, curve: torch.Tensor) -> torch.Tensor:
        dh = curve[:, :, :, 1:] - curve[:, :, :, :-1]
        dv = curve[:, :, 1:, :] - curve[:, :, :-1, :]
        return (dh.abs().mean() + dv.abs().mean())


class ZeroDCELoss(nn.Module):
    def __init__(self, weights: tuple[float, float, float, float] = (1.0, 10.0, 5.0, 200.0)) -> None:
        super().__init__()
        self.spatial = SpatialConsistencyLoss()
        self.exposure = ExposureControlLoss()
        self.color = ColorConstancyLoss()
        self.smooth = IlluminationSmoothnessLoss()
        self.weights = weights

    def forward(self, enhanced: torch.Tensor, original: torch.Tensor, curve: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {
            "spatial": self.spatial(enhanced, original),
            "exposure": self.exposure(enhanced),
            "color": self.color(enhanced),
            "smoothness": self.smooth(curve),
        }
        total = sum(weight * losses[name] for weight, name in zip(self.weights, losses))
        losses["total"] = total
        return losses
