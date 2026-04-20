"""
TransUNetPS — TransUNet-based architecture for Uncalibrated Photometric Stereo.

Closely follows the TransUNet implementation from:
    Chen et al., "TransUNet: Rethinking the U-Net architecture design for medical
    image segmentation through the lens of transformers", Medical Image Analysis, 2024.
    Source: https://github.com/Beckschen/TransUNet

Architecture (Encoder-only configuration):
    Part I:   CNN Encoder (shared across N images for photometric stereo)
    Fusion:   Multi-scale max-pooling across N images (our contribution)
    Part III: Transformer Encoder (Attention + MLP blocks, following TransUNet exactly)
    Part II:  CNN Decoder (Cascaded upsampler with skip connections)
    Head:     Normal prediction (L2-normalized) or segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# Transformer Components (following TransUNet vit_seg_modeling.py)

class Attention(nn.Module):
    """
    Multi-head self-attention (TransUNet vit_seg_modeling.py:50-94).
    """

    def __init__(self, hidden_size, num_heads, attention_dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

    def _transpose_for_scores(self, x):
        # (B, seq_len, all_head_size) -> (B, num_heads, seq_len, head_size)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_size)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        output = self.out(context)
        output = self.proj_dropout(output)
        return output


class Mlp(nn.Module):
    """
    MLP block (TransUNet vit_seg_modeling.py:97-119).
    fc1 ---> GELU ---> dropout ---> fc2 ---> dropout
    """

    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-LayerNorm (TransUNet vit_seg_modeling.py:168-224).
        z' = MSA(LN(z)) + z
        z  = MLP(LN(z')) + z'
    """

    def __init__(self, hidden_size, num_heads, mlp_dim,
                 dropout_rate=0.1, attention_dropout_rate=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h  # residual

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # residual
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer blocks + final LayerNorm (TransUNet vit_seg_modeling.py:227-244).
    """

    def __init__(self, hidden_size, num_heads, mlp_dim, num_layers,
                 dropout_rate=0.1, attention_dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.encoder_norm(hidden_states)


class Embeddings(nn.Module):
    """
    Patch embedding + positional encoding (TransUNet vit_seg_modeling.py:122-165).
    Tokenizes CNN feature maps into a sequence with learned position embeddings.
    """

    def __init__(self, in_channels, hidden_size, img_size=8, patch_size=1, dropout_rate=0.1):
        super().__init__()
        # Patch embedding via conv projection
        self.patch_embeddings = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )
        n_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.patch_size = patch_size

    def forward(self, x):
        """x: (B, C, H, W) -> (B, n_patches, hidden_size)"""
        x = self.patch_embeddings(x)  # (B, hidden_size, H', W')
        self._h, self._w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden_size)

        # Add position embeddings (interpolate if spatial size changed)
        seq_len = x.shape[1]
        if seq_len != self.position_embeddings.shape[1]:
            pos_embed = self._interpolate_pos_embed(self._h, self._w)
        else:
            pos_embed = self.position_embeddings

        x = x + pos_embed
        x = self.dropout(x)
        return x

    def _interpolate_pos_embed(self, target_h, target_w):
        """Interpolate position embeddings for different spatial sizes."""
        pos = self.position_embeddings  # (1, orig_len, hidden)
        hidden = pos.shape[2]
        orig_len = pos.shape[1]
        orig_h = orig_w = int(math.sqrt(orig_len))

        pos = pos.reshape(1, orig_h, orig_w, hidden).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(target_h, target_w),
                            mode='bilinear', align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, hidden)
        return pos


# ══════════════════════════════════════════════════════════════════════
# CNN Encoder (Part I) — shared across N images for photometric stereo
# ══════════════════════════════════════════════════════════════════════

class Conv2dReLU(nn.Sequential):
    """Conv2d + BatchNorm + ReLU (TransUNet vit_seg_modeling.py:259-281)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    """Two Conv2dReLU layers (U-Net style encoder block)."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch, out_ch)
        self.conv2 = Conv2dReLU(out_ch, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dropout(x)


class SharedEncoder(nn.Module):
    """
    Shared-weight CNN encoder (Part I), applied per image.
    Returns multi-scale features for skip connections.
    """

    def __init__(self, in_channels=1, channels=(64, 128, 256, 512), dropout=0.1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, channels[0], dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EncoderBlock(channels[0], channels[1], dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = EncoderBlock(channels[1], channels[2], dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = EncoderBlock(channels[2], channels[3], dropout)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)          # (B*N, ch[0], H, W)
        e2 = self.enc2(self.pool1(e1))  # (B*N, ch[1], H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B*N, ch[2], H/4, W/4)
        e4 = self.enc4(self.pool3(e3))  # (B*N, ch[3], H/8, W/8)
        bottleneck = self.pool4(e4)     # (B*N, ch[3], H/16, W/16)
        return [e1, e2, e3, e4], bottleneck


# Multi-Scale Fusion (our contribution for photometric stereo)

class MultiScaleFusion(nn.Module):
    """Max-pool across N images at each encoder scale + bottleneck."""

    def forward(self, skip_features_list, bottleneck, B, N, image_counts=None):
        """
        Args:
            skip_features_list: list of (B*N, C, H, W) at each encoder scale
            bottleneck: (B*N, C, H/16, W/16)
            B, N: batch size, max images
            image_counts: (B,) actual counts
        Returns:
            fused_skips: list of (B, C, H, W) at each scale
            fused_bottleneck: (B, C, H/16, W/16)
        """
        all_features = skip_features_list + [bottleneck]
        fused = []
        for feat in all_features:
            C, fH, fW = feat.shape[1], feat.shape[2], feat.shape[3]
            feat = feat.view(B, N, C, fH, fW)

            if image_counts is not None:
                mask = torch.arange(N, device=feat.device).unsqueeze(0) < image_counts.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                feat = torch.where(mask, feat, torch.tensor(float("-inf"), device=feat.device))

            pooled, _ = feat.max(dim=1)
            fused.append(pooled)

        return fused[:-1], fused[-1]  # skips, bottleneck


# CNN Decoder (Part II) — following TransUNet DecoderCup

class DecoderBlock(nn.Module):
    """
    Decoder block (TransUNet vit_seg_modeling.py:284-315).
    Upsample ---> Cat(skip) ---> Conv2dReLU × 2
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch)
        self.conv2 = Conv2dReLU(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class DecoderCup(nn.Module):
    """
    Cascaded upsampler decoder (TransUNet vit_seg_modeling.py:326-367).

    Transforms Transformer output back to spatial features and
    progressively upsamples with skip connections from CNN encoder.
    """

    def __init__(self, hidden_size, encoder_channels, decoder_channels, n_skip=4):
        super().__init__()
        head_ch = decoder_channels[0]
        self.conv_more = Conv2dReLU(hidden_size, head_ch)

        self.blocks = nn.ModuleList()
        in_ch = head_ch
        for i, out_ch in enumerate(decoder_channels[1:]):
            skip_ch = encoder_channels[-(i + 1)] if i < n_skip else 0
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.n_skip = n_skip

    def forward(self, hidden_states, skips, h, w):
        B = hidden_states.shape[0]
        # Reshape tokens to spatial: (B, seq_len, hidden) ---> (B, hidden, h, w)
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, -1, h, w)
        x = self.conv_more(x)

        # Reverse skips for decoder (highest resolution last ---> first)
        skips = list(reversed(skips)) if skips else []

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) and i < self.n_skip else None
            x = block(x, skip)

        return x


# Output Heads

class NormalHead(nn.Module):
    """Normal map prediction: Conv ---> L2 normalize."""

    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return F.normalize(x, dim=1, eps=1e-8)


class SegmentationHead(nn.Module):
    """Segmentation head (TransUNet vit_seg_modeling.py:318-323)."""

    def __init__(self, in_ch, num_classes, upscale_factor=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if upscale_factor > 1 else nn.Identity()

    def forward(self, x):
        return self.up(self.conv(x))


# TransUNetPS — Main Model

class TransUNetPS(nn.Module):
    """
    TransUNet for Photometric Stereo (and segmentation).

    Closely follows the TransUNet Encoder-only architecture with our
    additions for multi-image photometric stereo.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            from config import ModelConfig
            config = ModelConfig()

        enc_ch = tuple(config.encoder_channels)
        dec_ch = tuple(config.decoder_channels)
        hidden_size = config.hidden_size
        self.mode = getattr(config, 'mode', 'normal')

        # [QUAN TRỌNG]: ÉP CỨNG ĐẦU VÀO = 1 KÊNH (ẢNH XÁM)
        self.encoder = SharedEncoder(
            in_channels=getattr(config, 'in_channels', 3),
            channels=enc_ch,
            dropout=config.dropout,
        )

        # Multi-scale fusion (for photometric stereo with N images)
        self.fusion = MultiScaleFusion()

        # Patch embedding: CNN bottleneck ---> Transformer tokens
        self.embeddings = Embeddings(
            in_channels=enc_ch[-1],  # bottleneck channels
            hidden_size=hidden_size,
            img_size=8,  # H/16 for 128 patch, adaptable
            patch_size=1,
            dropout_rate=config.transformer_dropout,
        )

        # Part III: Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            hidden_size=hidden_size,
            num_heads=config.transformer_heads,
            mlp_dim=config.transformer_ff_dim,
            num_layers=config.transformer_layers,
            dropout_rate=config.transformer_dropout,
            attention_dropout_rate=config.attention_dropout_rate,
        )

        # Part II: CNN Decoder
        self.decoder = DecoderCup(
            hidden_size=hidden_size,
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            n_skip=config.n_skip,
        )

        # Output head
        if self.mode == 'segmentation':
            self.head = SegmentationHead(dec_ch[-1], config.num_classes)
        else:
            self.head = NormalHead(dec_ch[-1], config.out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, image_counts=None):
        if images.dim() == 5:
            # Photometric stereo mode: (B, N, C, H, W)
            B, N, C, H, W = images.shape
            x = images.view(B * N, C, H, W)
            skips, bottleneck = self.encoder(x)
            fused_skips, fused_bn = self.fusion(skips, bottleneck, B, N, image_counts)
        else:
            # Single image mode: (B, C, H, W) — for segmentation
            B, C, H, W = images.shape
            skips, bottleneck = self.encoder(images)
            fused_skips = skips
            fused_bn = bottleneck

        # Patch embedding + Transformer encoder
        tokens = self.embeddings(fused_bn)
        h, w = self.embeddings._h, self.embeddings._w
        encoded = self.transformer_encoder(tokens)

        # Decoder with skip connections
        decoded = self.decoder(encoded, fused_skips, h, w)

        # Output
        return self.head(decoded)


# LightweightUNetPS — Previous lightweight architecture (~4.7M params)

class _LWConvBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.block(x))


class _LWDownBlock(nn.Module):
    """MaxPool -> ConvBlock."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _LWConvBlock(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class _LWUpBlock(nn.Module):
    """Upsample -> Concat skip -> ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _LWConvBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightweightUNetPS(nn.Module):
    """
    Lightweight U-Net + Transformer for Uncalibrated Photometric Stereo.
    """

    # Default config for lightweight model
    DEFAULTS = dict(
        in_channels=1,
        encoder_channels=[32, 64, 128, 256],
        bottleneck_channels=256,
        decoder_channels=[128, 64, 32, 32],
        out_channels=3,
        transformer_heads=4,
        transformer_ff_dim=512,
        transformer_layers=2,
        transformer_dropout=0.1,
        dropout=0.1,
    )

    def __init__(self, config=None):
        super().__init__()
        d = self.DEFAULTS.copy()
        if config is not None:
            # Only take non-architecture settings from config
            for k in ['in_channels', 'out_channels', 'dropout',
                       'transformer_heads', 'transformer_ff_dim',
                       'transformer_layers', 'transformer_dropout']:
                if hasattr(config, k):
                    d[k] = getattr(config, k)

        # Sử dụng in_channels từ config
        if config is not None and hasattr(config, 'in_channels'):
            d['in_channels'] = config.in_channels

        enc_ch = d['encoder_channels']
        dec_ch = d['decoder_channels']
        bn_ch = d['bottleneck_channels']

        # Encoder
        self.enc1 = _LWConvBlock(d['in_channels'], enc_ch[0], d['dropout'])
        self.enc2 = _LWDownBlock(enc_ch[0], enc_ch[1], d['dropout'])
        self.enc3 = _LWDownBlock(enc_ch[1], enc_ch[2], d['dropout'])
        self.enc4 = _LWDownBlock(enc_ch[2], enc_ch[3], d['dropout'])
        self.bottleneck = _LWDownBlock(enc_ch[3], bn_ch, d['dropout'])

        # Fusion
        self.fusion = MultiScaleFusion()

        # Transformer bottleneck (PyTorch built-in)
        self.patch_proj = nn.Conv2d(bn_ch, bn_ch, 1)
        self.pos_embed_row = nn.Embedding(64, bn_ch // 2)
        self.pos_embed_col = nn.Embedding(64, bn_ch // 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bn_ch,
            nhead=d['transformer_heads'],
            dim_feedforward=d['transformer_ff_dim'],
            dropout=d['transformer_dropout'],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=d['transformer_layers'])
        self.transformer_norm = nn.LayerNorm(bn_ch)
        self.proj_back = nn.Conv2d(bn_ch, bn_ch, 1)

        # Decoder
        self.up4 = _LWUpBlock(bn_ch, enc_ch[3], dec_ch[0], d['dropout'])
        self.up3 = _LWUpBlock(dec_ch[0], enc_ch[2], dec_ch[1], d['dropout'])
        self.up2 = _LWUpBlock(dec_ch[1], enc_ch[1], dec_ch[2], d['dropout'])
        self.up1 = _LWUpBlock(dec_ch[2], enc_ch[0], dec_ch[3], d['dropout'])

        # Head
        self.head = nn.Conv2d(dec_ch[3], d['out_channels'], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)
        return [e1, e2, e3, e4], bn

    def _add_pos_encoding(self, x):
        H, W = x.shape[2], x.shape[3]
        row_pos = self.pos_embed_row(torch.arange(H, device=x.device))
        col_pos = self.pos_embed_col(torch.arange(W, device=x.device))
        pos = torch.cat([
            row_pos.unsqueeze(1).expand(-1, W, -1),
            col_pos.unsqueeze(0).expand(H, -1, -1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)
        return x + pos

    def forward(self, images, image_counts=None):
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            x = images.view(B * N, C, H, W)
            skips, bn = self._encode(x)
            all_feats = skips + [bn]
            fused = []
            for feat in all_feats:
                Cf, fH, fW = feat.shape[1], feat.shape[2], feat.shape[3]
                feat = feat.view(B, N, Cf, fH, fW)
                if image_counts is not None:
                    mask = torch.arange(N, device=feat.device).unsqueeze(0) < image_counts.unsqueeze(1)
                    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    feat = torch.where(mask, feat, torch.tensor(float("-inf"), device=feat.device))
                pooled, _ = feat.max(dim=1)
                fused.append(pooled)
            fused_skips = fused[:4]
            fused_bn = fused[4]
        else:
            B, C, H, W = images.shape
            skips, bn = self._encode(images)
            fused_skips = skips
            fused_bn = bn

        # Transformer bottleneck
        x = self.patch_proj(fused_bn)
        residual = x
        x = self._add_pos_encoding(x)
        Bt, Ct, Ht, Wt = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.transformer(x)
        x = self.transformer_norm(x)
        x = x.permute(0, 2, 1).view(Bt, Ct, Ht, Wt)
        x = self.proj_back(x) + residual

        # Decoder
        x = self.up4(x, fused_skips[3])
        x = self.up3(x, fused_skips[2])
        x = self.up2(x, fused_skips[1])
        x = self.up1(x, fused_skips[0])

        # Normal output
        x = self.head(x)
        return F.normalize(x, dim=1, eps=1e-8)


#======
# Factory function — selects model by type
#======

# Backward compatibility alias
UNetPS = TransUNetPS

def get_model(config=None, model_type="transunet"):
    """
    Factory function to create the model.
    """
    if model_type == "lightweight":
        return LightweightUNetPS(config)
    else:
        return TransUNetPS(config)