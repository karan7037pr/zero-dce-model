# Zero-DCE low-light enhancement

This repository implements Zero-DCE (Zero-Reference Deep Curve Estimation) in PyTorch. It learns pixel-wise enhancement curves directly from low-light images, so training does not require paired normal-light targets.

The implementation follows the report supplied for this project and the original method: DCE-Net predicts eight 24-channel curve maps, the enhancement curve is applied iteratively, and training combines spatial consistency, exposure control, color constancy, and illumination smoothness losses.

Report: [karan VLG report 22113067.pdf](https://drive.google.com/file/d/1LWyCtH2upRaVs20Nzn-SM9vB9FqBjPpw/view)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Put low-light images below `data/` (subdirectories are supported):

```bash
python train.py data --epochs 50 --batch-size 8 --output checkpoints/zerodce.pt
```

## Enhance

```bash
python enhance.py input.jpg enhanced.jpg --checkpoint checkpoints/zerodce.pt
python enhance.py low_light_folder enhanced_folder --checkpoint checkpoints/zerodce.pt
```

## Tests

```bash
pytest -q
```

The model is intentionally kept at the original 32-channel width and uses no reference images or pretrained backbone.
