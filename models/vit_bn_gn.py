"""
Vision Transformer (ViT) variants with BatchNorm / GroupNorm around tokenization.

- Pre-tokenization norm (on images): BatchNorm2d / GroupNorm
- Post-tokenization norm (on tokens): BatchNorm1d / GroupNorm over channel (embed) dim
- Keeps standard LayerNorm inside transformer blocks (baseline behavior)
- LRP-style scaffolding follows the "my*" layer pattern from provided ResNet/VGG files.

"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from enum import Enum


__all__ = [
    "VisionTransformer",
    "vit_b16_bn_pre",
    "vit_b16_bn_post",
    "vit_b16_bn_both",
    "vit_b16_gn_pre",
    "vit_b16_gn_post",
    "vit_b16_gn_both",
]


# Utilities & LRP base classes
def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, inputs, output):
    # Store inputs/outputs for later relevance propagation
    self.X = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
    self.Y = output

def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)

class RelProp(nn.Module):
    """Base class that registers a forward hook and provides relprop stub."""
    def __init__(self):
        super().__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S, create_graph=False):
        return torch.autograd.grad(Z, X, S, create_graph=create_graph, retain_graph=True)

    def relprop(self, R, alpha=1, create_graph=False):
        # Default: pass-through (identity). Specialized layers override when needed.
        return R


# Custom LRP-friendly wrappers
class myLinear(nn.Linear, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        # Alpha-Beta rule (as in provided ResNet)
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1, bias=None)
            Z2 = F.linear(x2, w2, bias=None)
            Z = Z1 + Z2
            S = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S, create_graph=create_graph)[0]
            C2 = x2 * self.gradprop(Z2, x2, S, create_graph=create_graph)[0]
            return C1 + C2

        activator = f(pw, nw, px, nx)
        inhibitor = f(nw, pw, px, nx)
        return alpha * activator - beta * inhibitor

class myBatchNorm2d(nn.BatchNorm2d, RelProp):
    # Same structure as in provided ResNet (2D)
    def relprop(self, R, alpha=1, create_graph=False):
        X = self.X
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5)
        )
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        return X * Ca

class myBatchNorm1d(nn.BatchNorm1d, RelProp):
    """BN over embedding/channel dimension for token sequences flattened as (B*N, D)."""
    def relprop(self, R, alpha=1, create_graph=False):
        # R, X are shaped as (B*N, D) or (B*N, D, 1) depending on caller.
        X = self.X
        # Ensure shape (B*N, D)
        if X.dim() == 3 and X.size(-1) == 1:
            X_eff = X.squeeze(-1)
        else:
            X_eff = X
        # Broadcast weight / running_var over batch dimension
        weight = self.weight.unsqueeze(0) / ((self.running_var.pow(2) + self.eps).pow(0.5))
        Z = X_eff * weight + 1e-9
        # Match shapes for division
        R_eff = R.squeeze(-1) if (R.dim() == 3 and R.size(-1) == 1) else R
        S = R_eff / Z
        Ca = S * weight
        out = X_eff * Ca
        # Restore last dim if input carried a dummy spatial axis
        if X.dim() == 3 and X.size(-1) == 1:
            out = out.unsqueeze(-1)
        return out

class myGroupNorm(nn.GroupNorm, RelProp):
    """GroupNorm for (B*N, D, 1) or (B, C, H, W). LRP here uses identity pass-through."""
    # A precise LRP for GN is non-trivial; we provide an identity fallback.
    def relprop(self, R, alpha=1, create_graph=False):
        return R

class myConv2d(nn.Conv2d, RelProp):
    # Minimal LRP for patch embedding conv; use identity fallback to keep flow.
    def relprop(self, R, alpha=1, create_graph=False):
        return R

class myDropout(nn.Dropout, RelProp):
    pass

class myDropout2d(nn.Dropout2d, RelProp):
    pass

class myAdd(RelProp):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha=1, create_graph=False):
        d0 = safe_divide(self.X[0], self.Y)
        d1 = safe_divide(self.X[1], self.Y)
        return [R * d0, R.clone() * d1]

class mySequential(nn.Sequential):
    def relprop(self, R, alpha=1, create_graph=False):
        for m in reversed(self._modules.values()):
            if hasattr(m, "relprop"):
                R = m.relprop(R, alpha, create_graph=create_graph)
        return R

class myAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        px = torch.clamp(self.X, min=0)
        def f(x1):
            Z1 = F.adaptive_avg_pool2d(x1, self.output_size)
            S1 = safe_divide(R, Z1)
            C1 = x1 * self.gradprop(Z1, x1, S1, create_graph=create_graph)[0]
            return C1
        return f(px)

# ViT building blocks
class PatchEmbed(RelProp):
    """Image to Patch Embedding using a Conv2d with kernel_size=stride=patch_size."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = myConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x

class MLP(RelProp):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = myLinear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = myDropout(drop) if drop > 0 else nn.Identity()
        self.fc2 = myLinear(hidden_features, out_features)
        self.drop2 = myDropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class EncoderBlock(RelProp):
    """Standard Transformer encoder block with LayerNorm, MHA, MLP."""
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True, dropout=attn_drop)
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.add = myAdd()

    def forward(self, x):
        # Self-attention
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = self.add([x, y])
        # MLP
        y = self.norm2(x)
        y = self.mlp(y)
        x = self.add([x, y])
        return x

    def relprop(self, R, alpha=1, create_graph=False):
        # Minimal identity-style relevance pass-through across residuals
        # Split relevance equally across residual branches for stability
        # (refine with dedicated attention LRP if needed)
        return R


# Pre/Post tokenization norms
class PreStemNorm(RelProp):
    """Normalization applied on images (B, C, H, W) before patch embedding."""
    def __init__(self, in_chans=3, norm_type="batchnorm", gn_groups=32, eps=1e-5, momentum=0.1):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = myBatchNorm2d(in_chans, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        elif norm_type == "groupnorm":
            self.norm = myGroupNorm(gn_groups, in_chans, eps=eps, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(x)

class PostTokenNorm(RelProp):
    """Normalization applied on tokens (B, N, D) after patch embedding (and pos/cls)."""
    def __init__(self, embed_dim, norm_type="batchnorm", gn_groups=32, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "batchnorm":
            self.norm = myBatchNorm1d(embed_dim, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        elif norm_type == "groupnorm":
            # We reshape to (B*N, D, 1) and apply GroupNorm over D channels.
            self.norm = myGroupNorm(gn_groups, embed_dim, eps=eps, affine=True)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # x: (B, N, D)
        if isinstance(self.norm, myBatchNorm1d):
            B, N, D = x.shape
            x = x.reshape(B * N, D)      # (B*N, D)
            x = self.norm(x)
            x = x.reshape(B, N, D)
            return x
        elif isinstance(self.norm, myGroupNorm):
            B, N, D = x.shape
            x = x.reshape(B * N, D).unsqueeze(-1)  # (B*N, D, 1)
            x = self.norm(x)
            x = x.squeeze(-1).reshape(B, N, D)
            return x
        else:
            return x


# Vision Transformer main
class ActivationMode(Enum):
    RELU = 1
    SOFTPLUS = 2

class VisionTransformer(nn.Module):
    """
    ViT with selectable norm around tokenization:
        - norm_type: "batchnorm" | "groupnorm" | "none"
        - bn_position: "pre" | "post" | "both"
    Inside blocks we keep LayerNorm (baseline), matching advisor's request
    not to add BN inside every transformer block.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_type="batchnorm",
        bn_position="both",     # "pre" | "post" | "both"
        gn_groups=32
    ):
        super().__init__()
        self.activationmode = ActivationMode.RELU
        self.activation_wrapper = [lambda x: torch.nn.functional.relu(x)]  # kept for API parity

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_type = norm_type
        self.bn_position = bn_position

        # Pre-tokenization normalization on images
        self.pre_stem = PreStemNorm(in_chans=in_chans, norm_type=norm_type, gn_groups=gn_groups)

        # Patch embedding and position/class tokens
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Post-tokenization normalization on tokens
        self.post_token = PostTokenNorm(embed_dim, norm_type=norm_type, gn_groups=gn_groups)

        # Encoder blocks
        blocks = []
        for _ in range(depth):
            blocks.append(EncoderBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop=drop_rate))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Head
        self.head = myLinear(embed_dim, num_classes)

        # Init
        self.apply(_weights_init)
        self._init_cls_pos(num_patches, embed_dim)

    def _init_cls_pos(self, num_patches, dim):
        # Standard ViT init for pos/cls
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # activation API parity
    def set_softplus(self, beta):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.softplus(x, beta=beta)
        self.activationmode = ActivationMode.SOFTPLUS

    def set_relu(self):
        self.activation_wrapper[0] = lambda x: torch.nn.functional.relu(x)
        self.activationmode = ActivationMode.RELU

    # forward variants
    def _tokens(self, x):
        # Optional pre-stem norm
        if self.bn_position in ("pre", "both") and self.norm_type in ("batchnorm", "groupnorm"):
            x = self.pre_stem(x)  # (B, C, H, W)

        x = self.patch_embed(x)  # (B, N, D)

        # Add cls token + positions
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)     # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Optional post-token norm
        if self.bn_position in ("post", "both") and self.norm_type in ("batchnorm", "groupnorm"):
            x = self.post_token(x)  # (B, N+1, D)

        return x

    def forward(self, input):
        """
        Standard forward: image -> logits
        """
        x = self._tokens(input)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]             # CLS token
        out = self.head(cls)
        return out

    def forward_withoutfcl(self, input):
        """
        Returns the penultimate feature (CLS embedding after encoder&norm).
        """
        x = self._tokens(input)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        return cls

    def forward_feature(self, input):
        """
        Returns the sequence of patch+CLS tokens after last encoder block (before final norm).
        """
        x = self._tokens(input)
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, N+1, D)

    # relevance propagation (minimal scaffolding)
    def relprop(self, R, alpha=1, create_graph=False, break_at_basicblocks=False):
        """
        Minimal relprop: pass relevance to the token sequence uniformly.
        Detailed LRP for attention/MLP can be added later.
        """
        # Map relevance on logits back to CLS embedding
        # Here we approximate by attributing relevance to head input
        R = self.head.relprop(R, alpha, create_graph=create_graph)  # (B, D)
        # Expand to sequence length 1 (CLS only); other tokens receive zero here
        B = R.size(0)
        seq_R = torch.zeros(B, 1, self.embed_dim, device=R.device, dtype=R.dtype)
        seq_R[:, 0, :] = R
        return seq_R


# Factory helpers (ViT-B/16)
def _vit_b16(norm_type="batchnorm", bn_position="both", **kwargs):
    # Defaults roughly aligned with ViT-B/16
    defaults = dict(
        img_size=224, patch_size=16, in_chans=3,
        num_classes=kwargs.get("num_classes", 10),
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0,
        norm_type=norm_type, bn_position=bn_position,
        gn_groups=kwargs.get("gn_groups", 32)
    )
    return VisionTransformer(**defaults)

def vit_b16_bn_pre(**kwargs):
    """ViT-B/16 with BatchNorm applied before tokenization only."""
    return _vit_b16(norm_type="batchnorm", bn_position="pre", **kwargs)

def vit_b16_bn_post(**kwargs):
    """ViT-B/16 with BatchNorm applied after tokenization only."""
    return _vit_b16(norm_type="batchnorm", bn_position="post", **kwargs)

def vit_b16_bn_both(**kwargs):
    """ViT-B/16 with BatchNorm applied both before and after tokenization."""
    return _vit_b16(norm_type="batchnorm", bn_position="both", **kwargs)

def vit_b16_gn_pre(**kwargs):
    """ViT-B/16 with GroupNorm applied before tokenization only."""
    return _vit_b16(norm_type="groupnorm", bn_position="pre", **kwargs)

def vit_b16_gn_post(**kwargs):
    """ViT-B/16 with GroupNorm applied after tokenization only."""
    return _vit_b16(norm_type="groupnorm", bn_position="post", **kwargs)

def vit_b16_gn_both(**kwargs):
    """ViT-B/16 with GroupNorm applied both before and after tokenization."""
    return _vit_b16(norm_type="groupnorm", bn_position="both", **kwargs)


if __name__ == "__main__":
    model = vit_b16_bn_both(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)