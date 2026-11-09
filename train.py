"""Train Zero-DCE without paired or reference-enhanced images."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from zerodce.data import LowLightDataset
from zerodce.losses import ZeroDCELoss
from zerodce.model import DCEModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory containing low-light images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/zerodce.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    loader = DataLoader(LowLightDataset(args.data, args.image_size), args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = DCEModel().to(device)
    criterion = ZeroDCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images in loader:
            images = images.to(device, non_blocking=True)
            enhanced, curve = model(images)
            losses = criterion(enhanced, images, curve)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            running += losses["total"].item()
        mean_loss = running / len(loader)
        print(f"epoch {epoch:03d}/{args.epochs:03d} loss={mean_loss:.5f}")
        torch.save({"model": model.state_dict(), "epoch": epoch}, args.output)


if __name__ == "__main__":
    main()
