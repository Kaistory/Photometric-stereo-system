"""
Configuration for TransUNetPS.

We did it to align with TransUNet (Chen et al., Medical Image Analysis, 2024).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    data_root: str = "./data/training"
    objects: List[str] = field(default_factory=lambda: [
        "ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
        "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG",
    ])
    num_lights: int = 96
    original_h: int = 512
    original_w: int = 612

    # Training patches
    patch_size: int = 128
    patches_per_epoch: int = 2000
    num_input_images: int = 32
    min_input_images: int = 8

    # Dataloader
    batch_size: int = 4
    num_workers: int = 0  # 0 for macOS compatibility


@dataclass
class ModelConfig:
    """
    TransUNetPS model configuration.

    Aligned with TransUNet but lightweight for graduation project.
    TransUNet original: hidden=768, heads=12, layers=12, mlp=3072 (~86M params)
    Ours: hidden=256, heads=4, layers=2, mlp=512 (~5M params, lightweight)
    """
    in_channels: int = 3  # grayscale input
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 16])
    out_channels: int = 3  # normal map (x, y, z)

    # Transformer encoder (following TransUNet)
    hidden_size: int = 256  # TransUNet: 768, ours lightweight: 256
    transformer_heads: int = 4  # TransUNet: 12, ours: 4
    transformer_ff_dim: int = 512  # TransUNet: 3072, ours: 512
    transformer_layers: int = 2  # TransUNet: 12, ours: 2
    transformer_dropout: float = 0.1  # Same as TransUNet
    attention_dropout_rate: float = 0.0  # Same as TransUNet

    # Decoder skip connections
    n_skip: int = 4  # TransUNet uses 3 for R50, we use 4 (all encoder levels)

    # Regularization
    dropout: float = 0.1

    # Lightweight model defaults (used when model_type="lightweight")
    bottleneck_channels: int = 256

    # Segmentation mode (for Synapse)
    mode: str = "normal"  # "normal" for photometric stereo, "segmentation" for Synapse
    num_classes: int = 9  # Synapse: 9 organ classes

    # Model selection
    model_type: str = "transunet"  # "transunet" (~11M) or "lightweight" (~4.7M)


@dataclass
class TrainConfig:
    """
    Training configuration.

    For photometric stereo (training from scratch): AdamW, lr=1e-3, cosine annealing
    For Synapse (TransUNet alignment): SGD, lr=0.01, poly decay
    """
    epochs: int = 150
    lr: float = 1e-3  # AdamW default for training from scratch
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD (Synapse mode)
    lr_power: float = 0.9  # for poly decay (Synapse mode)
    optimizer: str = "adamw"  # "adamw" for photometric stereo, "sgd" for synapse

    # Loss weights (for photometric stereo mode)
    angular_weight: float = 0.5
    cosine_weight: float = 0.3
    l1_weight: float = 0.2

    # Early stopping
    patience: int = 50  # more patient (was 30, stopped too early)

    # Gradient clipping
    grad_clip: float = 1.0

    # Checkpointing
    save_dir: str = "./checkpoints"
    log_interval: int = 10


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
