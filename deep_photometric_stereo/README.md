# TransUNetPS — Uncalibrated Photometric Stereo with TransUNet

Predict surface normal maps from multiple images under **unknown lighting** — no light direction input needed.

Based on the **TransUNet** architecture, closely following the original implementation:
> Chen et al., "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers", *Medical Image Analysis*, 2024.
> https://doi.org/10.1016/j.media.2024.103280
> Source: https://github.com/Beckschen/TransUNet

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision numpy scipy pillow tifffile

# 2. Prepare data
cd deep_photometric_stereo
python prepare_data.py

# 3. Train (TransUNet model — default)
python train.py --mode logo_cv --epochs 150

# 3b. Train (Lightweight model — faster, smaller)
python train.py --mode logo_cv --epochs 200 --model_type lightweight

# 4. Predict normal map from new images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./my_images/ --output ./output/
```

---

## Model Architectures

Two model architectures are available, selectable via `--model_type`:

### `transunet` (default) — ~11M params

Closely follows the **TransUNet Encoder-only** configuration (Parts I + III + II). Custom Attention, Mlp, and Block classes matching the original TransUNet source code.

### `lightweight` — ~4.6M params

Our previous lighter architecture using PyTorch built-in `nn.TransformerEncoderLayer`, smaller encoder channels `[32, 64, 128, 256]`, learned 2D positional encoding, and a simpler U-Net decoder. Faster to train, lower memory.

| | `transunet` (default) | `lightweight` |
|---|---|---|
| **Parameters** | ~11M | ~4.6M |
| **Encoder channels** | [64, 128, 256, 512] | [32, 64, 128, 256] |
| **Transformer** | Custom Attention/Mlp/Block (from TransUNet paper) | `nn.TransformerEncoderLayer` (PyTorch built-in) |
| **Positional encoding** | Learned 1D (TransUNet style) | Learned 2D (row + column embeddings) |
| **Decoder** | DecoderCup (TransUNet cascaded upsampler) | Simple U-Net UpBlocks |
| **Residual around transformer** | No | Yes |
| **Best for** | Final training, paper alignment | Quick experiments, limited GPU |

```bash
# Select model type via CLI
python train.py --model_type transunet   # default, ~11M params
python train.py --model_type lightweight # ~4.6M params, faster
```

---

## TransUNetPS Architecture (default)

### Alignment with TransUNet Source Code

| Component | TransUNet (`vit_seg_modeling.py`) | Our Implementation (`model.py`) |
|---|---|---|
| **Attention** | `Attention` class (L50-94): Q/K/V projections, scaled dot-product | `Attention` class: identical design |
| **MLP** | `Mlp` class (L97-119): fc1 -> GELU -> dropout -> fc2 -> dropout | `Mlp` class: identical, Xavier init |
| **Block** | `Block` class (L168-224): pre-LayerNorm, MSA + residual, MLP + residual | `Block` class: identical pre-norm design |
| **Encoder** | `Encoder` class (L227-244): N x Block + final LayerNorm | `TransformerEncoder`: identical stack |
| **Embeddings** | `Embeddings` (L122-165): Conv patch embed + learned 1D pos embed | `Embeddings`: identical, with pos interpolation |
| **DecoderBlock** | `DecoderBlock` (L284-315): Upsample -> Cat(skip) -> Conv2dReLU x 2 | `DecoderBlock`: identical design |
| **DecoderCup** | `DecoderCup` (L326-367): conv_more head + cascaded decoder blocks | `DecoderCup`: identical, configurable n_skip |
| **Conv2dReLU** | `Conv2dReLU` (L259-281): Conv + BN + ReLU | `Conv2dReLU`: identical |
| **Training** | SGD(lr=0.01, momentum=0.9, wd=1e-4), poly LR decay (power=0.9) | Identical optimizer and scheduler |

### Our Additions for Photometric Stereo

| Addition | Purpose |
|---|---|
| **SharedEncoder** | Weight-shared CNN encoder processes each of N images independently |
| **MultiScaleFusion** | Max-pool across N images at every encoder scale (variable-N support) |
| **NormalHead** | L2-normalized 3-channel output for surface normals |
| **Dual mode** | Supports both `normal` (photometric stereo) and `segmentation` (Synapse) modes |
| **LightweightUNetPS** | Alternative smaller model option for fast experimentation |

```
Input: N grayscale images [B, N, 1, H, W]

Part I: CNN Encoder (shared weights across N images)
  enc1: 2xConv2dReLU(1->64)    -> [B*N, 64, H, W]
  enc2: Pool + 2xConv(64->128)  -> [B*N, 128, H/2, W/2]
  enc3: Pool + 2xConv(128->256) -> [B*N, 256, H/4, W/4]
  enc4: Pool + 2xConv(256->512) -> [B*N, 512, H/8, W/8]
  bottleneck: Pool               -> [B*N, 512, H/16, W/16]

Multi-Scale Fusion (our contribution):
  Max-pool across N at each level -> [B, C, h, w]

Part III: Transformer Encoder
  Patch Embedding: Conv 1x1 (512->256) + learned 1D pos embed
  2x TransformerBlock:
    z' = MSA(LN(z)) + z   (Eq.2 from TransUNet)
    z  = MLP(LN(z')) + z'  (Eq.3 from TransUNet)
  hidden=256, heads=4, mlp=512

Part II: CNN Decoder (DecoderCup)
  conv_more: Conv2dReLU(256->512)
  block1: Up + Cat(enc4) + 2xConv -> 256
  block2: Up + Cat(enc3) + 2xConv -> 128
  block3: Up + Cat(enc2) + 2xConv -> 64
  block4: Up + Cat(enc1) + 2xConv -> 16

NormalHead: Conv(16->3) -> L2 normalize -> [B, 3, H, W]
~11M parameters
```

---

## Comparison with Prior Work

| | TransUNet (original) | CNN-PS | SDM-UniPS | **TransUNetPS (ours)** |
|---|---|---|---|---|
| Task | Medical segmentation | Calibrated PS | Uncalibrated PS | **Uncalibrated PS** |
| Light dirs | N/A | Required | Not required | **Not required** |
| Architecture | CNN + ViT + U-Net | DenseNet | ConvNeXt+UPerNet+PMA | **SharedCNN + ViT + U-Net** |
| Params | ~41.4M (Enc+Dec) | ~80M | ~300M | **~11M / ~4.6M** |
| Transformer | 12-layer ViT, d=768 | None | Cross-image | **2-layer, d=256** |

---

## Training

### Hyperparameters (aligned with TransUNet)

| Parameter | TransUNet | Ours |
|---|---|---|
| Optimizer | SGD(momentum=0.9) | SGD(momentum=0.9) |
| Learning rate | 0.01 | 0.01 |
| LR schedule | Poly decay (power=0.9) | Poly decay (power=0.9) |
| Weight decay | 1e-4 | 1e-4 |
| Epochs | 150 | 150 |
| Grad clipping | N/A | max_norm=1.0 |

### Loss Functions

**Photometric Stereo mode:** `0.5 x Angular + 0.3 x Cosine + 0.2 x L1` (masked)

**Segmentation mode:** `0.5 x CrossEntropy + 0.5 x Dice` (same as TransUNet)

---

## Supported Datasets

| Dataset | Format | Description |
|---|---|---|
| **DiLiGenT** | `.mat`, `.png` | 10 objects x 96 images, 512x612 |
| **PRPS** | `.tif`, `gt_normal.tif` | Synthetic objects, specular/metallic variants |
| **Synapse** | `.npz`, `.h5` | 30 CT volumes, 9-class organ segmentation |

---

## Testing / Evaluation

Model type is **auto-detected** from the checkpoint — no need to pass `--model_type`.

```bash
# Prepare test data first (if not already done)
python prepare_data.py --data_root ./data/testing

# Test on ALL objects in a folder (recommended)
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing

# Test on ALL objects and save output images
python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing --save_output ./output/testing

# Test on a single object
python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG

# Test LOGO folds (if trained with --mode logo_cv)
python test.py --mode logo_eval --checkpoint_dir checkpoints/
```

`eval_all` prints a summary table:
```
============================================================
  Evaluation Summary — 10 objects
  Checkpoint: checkpoints/train/best.pt
============================================================
              ballPNG: 12.34
              bearPNG: 15.67
                  ...
            --------
              Average: 14.52
               Median: 13.80
                 Best: 12.34 (ballPNG)
                Worst: 18.90 (harvestPNG)
============================================================
```

---

## Prediction

Model type is **auto-detected** from the checkpoint.

```bash
# Predict normal map from a folder of images
python predict.py --checkpoint checkpoints/train/best.pt --input_dir ./data/testing/batteryPNG --output ./output/

# Predict with a mask
python predict.py --checkpoint checkpoints/train/best.pt --input_dir /path/to/images/ --mask /path/to/mask.png --output ./output/

# Predict from specific image files
python predict.py --checkpoint checkpoints/train/best.pt --images img1.png img2.png img3.png --output ./output/

# Output files:
#   output/predicted_normal.npy   (H, W, 3) float32
#   output/predicted_normal.png   RGB visualization
#   output/predicted_nx.png       X-component
#   output/predicted_ny.png       Y-component
#   output/predicted_nz.png       Z-component
```

---

## Project Structure

```
deep_photometric_stereo/
├── config.py           # Configuration (aligned with TransUNet)
├── model.py            # TransUNetPS (~11M) + LightweightUNetPS (~4.6M)
├── losses.py           # Masked angular + cosine + L1 losses
├── dataset.py          # DiLiGenT/PRPS/Synapse data loaders
├── train.py            # Training with SGD + poly LR (TransUNet style)
├── test.py             # Evaluation
├── predict.py          # Inference on new images -> normal map
├── prepare_data.py     # Data preparation (mat/tif/npz -> npy)
├── utils.py            # Metrics, checkpointing, visualization
├── data/               # Training data (gitignored)
│   ├── training/       # DiLiGenT: 10 objects x 96 images
│   ├── testing/        # DiLiGenT: 5 test objects
│   └── Synapse/        # Synapse: 2211 train slices + 12 test volumes (symlink)
└── README.md
```

---

## References

- **TransUNet**: Chen, J. et al. "TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers." *Medical Image Analysis* 97 (2024): 103280.
- **CNN-PS**: Ikehata, S. "CNN-PS: CNN-based photometric stereo for general non-convex surfaces." *ECCV* 2018.
- **SDM-UniPS**: Ikehata, S. "SDM-UniPS: Scalable, Detailed, and Mask-free Universal Photometric Stereo." *CVPR* 2023.
- **DiLiGenT**: Shi, B. et al. "A benchmark dataset and evaluation for non-Lambertian and uncalibrated photometric stereo." *IEEE TPAMI* 2019.
