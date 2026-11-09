"""Image-only dataset used by the zero-reference training loop."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class LowLightDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256) -> None:
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not self.files:
            raise FileNotFoundError(f"No images found under {root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            return TF.to_tensor(image)
