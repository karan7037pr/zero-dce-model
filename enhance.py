"""Enhance one image or every image in a directory with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from zerodce.model import DCEModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = DCEModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    files = [args.input] if args.input.is_file() else sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in files:
            with Image.open(path) as image:
                tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
            result, _ = model(tensor)
            destination = args.output / path.name if args.input.is_dir() else args.output
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_image(result[0].cpu(), destination)
            print(destination)


if __name__ == "__main__":
    main()
