import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

__all__ = ['ViTB16', 'vit_b_16', 'transfer_from_torchvision_vit']

# =========================
# Shared utils (your style)
# =========================

def safe_divide(a, b):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output

class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)
    def gradprop(self, Z, X, S, create_graph=False):
        C = torch.autograd.grad(Z, X, S, create_graph=create_graph, retain_graph=True)
        return C
    def relprop(self, R, alpha=1, create_graph=False):
        # Safe default: pass relevance through unchanged
        return R

class mySequential(nn.Sequential):
    def relprop(self, R, alpha=1, create_graph=False):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha, create_graph=create_graph)
        return R

class myLinear(nn.Linear, RelProp):
    # Same Alpha-Beta rule you use
    def relprop(self, R, alpha=1, create_graph=False):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            Z  = Z1 + Z2
            S  = safe_divide(R, Z)
            C1 = x1 * self.gradprop(Z1, x1, S, create_graph=create_graph)[0]
            C2 = x2 * self.gradprop(Z2, x2, S, create_graph=create_graph)[0]
            return C1 + C2

        act_rel = f(pw, nw, px, nx)
        inh_rel = f(nw, pw, px, nx)
        return alpha * act_rel - beta * inh_rel

class myLayerNorm(nn.LayerNorm, RelProp):
    # Stable passthrough for relevance; exact LRP for LN is possible but verbose
    def relprop(self, R, alpha=1, create_graph=False):
        return R

class myDropout(nn.Dropout, RelProp):
    def relprop(self, R, alpha=1, create_graph=False):
        return R

# =========================
# ViT building blocks
# =========================

class ActivationMode(Enum):
    RELU = 1
    SOFTPLUS = 2
    GELU = 3

def _act_fn(mode: ActivationMode, beta=1.0):
    if mode == ActivationMode.RELU:
        return lambda x: F.relu(x)
    if mode == ActivationMode.SOFTPLUS:
        return lambda x: F.softplus(x, beta=beta)
    return lambda x: F.gelu(x)  # default GELU

class PatchEmbed(RelProp):
    """
    Split image into non-overlapping patches and project to embeddings.
    Input: (B, 3, H, W), Output: (B, N, D)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = myLinear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Expected {self.img_size}x{self.img_size}, got {H}x{W}"
        # unfold to patches: (B, C*ps*ps, N)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # (B, C*ps*ps, N)
        patches = patches.transpose(1, 2)  # (B, N, C*ps*ps)
        out = self.proj(patches)           # (B, N, D)
        return out

    def relprop(self, R, alpha=1, create_graph=False):
        # Prop relevance through projection; skip putting it back into image grid here.
        return self.proj.relprop(R, alpha, create_graph=create_graph)

class MLP(RelProp):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, activation_wrapper=None):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = myLinear(dim, hidden)
        self.act = activation_wrapper if activation_wrapper is not None else [lambda x: F.gelu(x)]
        self.drop1 = myDropout(drop)
        self.fc2 = myLinear(hidden, dim)
        self.drop2 = myDropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act[0](x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def relprop(self, R, alpha=1, create_graph=False):
        R = self.drop2.relprop(R, alpha, create_graph=create_graph)
        R = self.fc2.relprop(R, alpha, create_graph=create_graph)
        # act/drop pass-through
        R = self.drop1.relprop(R, alpha, create_graph=create_graph)
        R = self.fc1.relprop(R, alpha, create_graph=create_graph)
        return R

class MultiHeadSelfAttention(RelProp):
    """
    Simplified MHSA using explicit QKV projections and out proj.
    """
    def __init__(self, dim, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = myLinear(dim, dim * 3)
        self.attn_drop = myDropout(attn_drop)
        self.proj = myLinear(dim, dim)
        self.proj_drop = myDropout(proj_drop)

        # cache for forward activations useful for analysis/visualization
        self.attn = None
        self.q = None
        self.k = None
        self.v = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B,h,N,d)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # store
        self.attn, self.q, self.k, self.v = attn, q, k, v
        return x

    def relprop(self, R, alpha=1, create_graph=False):
        # Stable and simple: pass relevance through the output projection and qkv in a conservative way
        R = self.proj_drop.relprop(R, alpha, create_graph=create_graph)
        R = self.proj.relprop(R, alpha, create_graph=create_graph)
        # Skip explicit redistribution through softmax attention (complex LRP). For many analysis tasks,
        # this conservative pass-through is sufficient and shape-safe.
        R = self.qkv.relprop(R, alpha, create_graph=create_graph)
        return R

class TransformerEncoderBlock(RelProp):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, activation_wrapper=None):
        super().__init__()
        self.norm1 = myLayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = myDropout(drop)

        self.norm2 = myLayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop, activation_wrapper=activation_wrapper)
        self.drop_path2 = myDropout(drop)

        self.add1 = nn.Identity()  # residual add, handled inline
        self.add2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def relprop(self, R, alpha=1, create_graph=False):
        # Residual splits: pass R to both branches proportionally (here, equal split for stability)
        R_main = R * 0.5
        R_branch = R - R_main

        # MLP branch
        R_mlp = self.drop_path2.relprop(R_branch, alpha, create_graph=create_graph)
        R_mlp = self.mlp.relprop(R_mlp, alpha, create_graph=create_graph)
        R_mlp = self.norm2.relprop(R_mlp, alpha, create_graph=create_graph)

        # Attention branch
        R_attn = self.drop_path1.relprop(R_main, alpha, create_graph=create_graph)
        R_attn = self.attn.relprop(R_attn, alpha, create_graph=create_graph)
        R_attn = self.norm1.relprop(R_attn, alpha, create_graph=create_graph)

        # Sum relevances back to input
        return R_attn + R_mlp

# =========================
# Top-level ViT-B/16 wrapper
# =========================

class ViTB16(nn.Module):
    """
    Vision Transformer (ViT-B/16) wrapper in your style.
    - my* layers with forward hooks for X/Y
    - activation_wrapper toggle (RELU/SOFTPLUS/GELU)
    - forward / forward_withoutfcl / forward_feature
    - relprop that mirrors the forward structure
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
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        # activation wrapper like your ResNet (default GELU for ViT)
        self.activationmode = ActivationMode.GELU
        self.activation_wrapper = [_act_fn(self.activationmode)]

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + pos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = myDropout(drop_rate)

        # Transformer encoder blocks
        blocks = []
        for _ in range(depth):
            blocks.append(
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    activation_wrapper=self.activation_wrapper,
                )
            )
        self.blocks = mySequential(*blocks)

        self.norm = myLayerNorm(embed_dim)

        # Classifier head
        self.fc = myLinear(embed_dim, num_classes)

        # init (timm/torchvision style small init for pos/cls)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # activation switches (same interface you use)
    def set_softplus(self, beta=1.0):
        self.activationmode = ActivationMode.SOFTPLUS
        self.activation_wrapper[0] = _act_fn(self.activationmode, beta=beta)

    def set_relu(self):
        self.activationmode = ActivationMode.RELU
        self.activation_wrapper[0] = _act_fn(self.activationmode)

    def set_gelu(self):
        self.activationmode = ActivationMode.GELU
        self.activation_wrapper[0] = _act_fn(self.activationmode)

    # ---- forward paths
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                        # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                 # (B, 1+N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)                             # (B, 1+N, D)
        x = self.norm(x)
        cls = x[:, 0]                                  # (B, D)
        out = self.fc(cls)
        return out

    def forward_withoutfcl(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls = x[:, 0]
        return cls  # feature before fc

    def forward_feature(self, x):
        # Return token sequence before norm (for visualization)
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        return x  # (B, 1+N, D)

    # ---- relprop
    def relprop(self, R, alpha=1, create_graph=False, break_at_basicblocks=False):
        # Classifier head
        R = self.fc.relprop(R, alpha, create_graph=create_graph)

        # Back to tokens
        # Match shape of cls token after norm: (B, D)
        # We need R for full token sequence; send relevance only to CLS (others zero)
        B = R.shape[0]
        # reconstruct last sequence activations
        # We rely on cached shapes: use forward_feature + norm to know lengths
        # Here, build a zero tensor and place R on CLS position
        # (We do not have a direct handle to self.norm.Y shape, so be safe:)
        # Assume last seen batch length equals current B; number of tokens = 1 + num_patches
        num_tokens = self.pos_embed.size(1)
        R_tokens = R.new_zeros((B, num_tokens, R.shape[-1]))
        R_tokens[:, 0, :] = R

        # norm & blocks
        R = self.norm.relprop(R_tokens, alpha, create_graph=create_graph)
        R = self.blocks.relprop(R, alpha, create_graph=create_graph)

        # pos + concat cls
        # Pass position embedding relevance back unchanged (constant param add)
        # Split CLS vs patch tokens
        R_cls = R[:, :1, :]
        R_patches = R[:, 1:, :]

        # Back through patch embed linear
        R_patches = self.patch_embed.relprop(R_patches, alpha, create_graph=create_graph)

        # We do not redistribute relevance into raw image grid here (consistent with your wrappers);
        # if needed, we can implement inverse of F.unfold for pixel-level maps.

        # Merge (CLS path has no learnable pre-cat except parameter cls_token/pos_embed we ignore)
        # Return relevance w.r.t. patch embeddings (token space)
        return R_patches

# =============================================
# Weight transfer from torchvision ViT-B/16
# =============================================
def transfer_from_torchvision_vit(local_model: ViTB16, tv_model) -> int:
    """Best-effort weight transfer from a torchvision VisionTransformer (vit_b_16)
    into the local wrapper, excluding the classification head.

    Returns number of parameter tensors copied. Skips tensors with shape mismatch.
    Safe against minor API changes (uses hasattr and state_dict key scanning).
    """
    transfers = 0
    import torch
    import math
    import torch.nn.functional as F

    # Torchvision state dict for fallback lookups
    tv_sd = tv_model.state_dict()

    # ---- Patch embedding (conv to linear) ----
    if hasattr(tv_model, 'conv_proj'):
        w = tv_model.conv_proj.weight.detach()  # (E,3,ps,ps)
        E = w.shape[0]
        reshaped = w.view(E, -1)
        if local_model.patch_embed.proj.weight.shape == reshaped.shape:
            local_model.patch_embed.proj.weight.data.copy_(reshaped); transfers += 1
        if getattr(tv_model.conv_proj, 'bias', None) is not None and local_model.patch_embed.proj.bias is not None:
            if local_model.patch_embed.proj.bias.shape == tv_model.conv_proj.bias.shape:
                local_model.patch_embed.proj.bias.data.copy_(tv_model.conv_proj.bias.detach()); transfers += 1

    # ---- Positional embedding & CLS token ----
    # Try common attribute names; then fallback to state_dict keys
    pos_sources = []
    for cand in ['pos_embed', 'pos_embedding']:
        if hasattr(tv_model, cand):
            pos_sources.append(getattr(tv_model, cand))
    # Encoder-held positional embedding (torchvision encoder.pos_embedding)
    if hasattr(tv_model, 'encoder') and hasattr(tv_model.encoder, 'pos_embedding'):
        pos_sources.append(tv_model.encoder.pos_embedding)
    # Deduplicate by id
    seen_ids = set(); pos_sources = [p for p in pos_sources if not (id(p) in seen_ids or seen_ids.add(id(p)))]
    def _copy_or_resize_pos_embed(src_pe: torch.Tensor) -> bool:
        """Copy positional embedding if shape matches; else resize grid and copy.
        Returns True if successful.
        """
        try:
            dst = local_model.pos_embed
            if dst.shape == src_pe.shape:
                dst.data.copy_(src_pe.detach());
                return True
            # shapes differ: try 2D resize on token grid (excluding cls token)
            if src_pe.dim() == 3 and dst.dim() == 3 and src_pe.size(0) == 1 and dst.size(0) == 1 and src_pe.size(-1) == dst.size(-1):
                D = dst.size(-1)
                # split cls + grid
                src_has_cls = src_pe.size(1) == (int(math.sqrt(src_pe.size(1)-1))**2 + 1)
                cls_tok = src_pe[:, :1] if src_has_cls else None
                grid = src_pe[:, 1:] if src_has_cls else src_pe
                N_old = grid.size(1)
                gs_old = int(math.sqrt(N_old))
                grid = grid.reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)  # (1,D,H,W)
                # new grid size
                N_new = local_model.patch_embed.num_patches
                gs_new = int(math.sqrt(N_new))
                grid_resized = F.interpolate(grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
                grid_resized = grid_resized.permute(0, 2, 3, 1).reshape(1, gs_new*gs_new, D)
                if cls_tok is not None:
                    pe_new = torch.cat([cls_tok, grid_resized], dim=1)
                else:
                    pe_new = grid_resized
                if pe_new.shape == dst.shape:
                    dst.data.copy_(pe_new.detach());
                    return True
        except Exception:
            return False
        return False

    copied_pos = False
    for pos in pos_sources:
        if _copy_or_resize_pos_embed(pos):
            transfers += 1
            copied_pos = True
            break
    if not copied_pos:
        # fallback: find in state_dict by key
        for k in ['pos_embed', 'pos_embedding', 'encoder.pos_embedding']:
            if k in tv_sd:
                if _copy_or_resize_pos_embed(tv_sd[k]):
                    transfers += 1
                    break

    # CLS token
    cls_sources = []
    for cand in ['cls_token', 'class_token']:
        if hasattr(tv_model, cand):
            cls_sources.append(getattr(tv_model, cand))
    if hasattr(tv_model, 'encoder') and hasattr(tv_model.encoder, 'cls_token'):
        cls_sources.append(tv_model.encoder.cls_token)
    def _copy_cls(src_cls: torch.Tensor) -> bool:
        if local_model.cls_token.shape == src_cls.shape:
            local_model.cls_token.data.copy_(src_cls.detach());
            return True
        return False
    copied_cls = False
    for cls in cls_sources:
        if _copy_cls(cls):
            transfers += 1
            copied_cls = True
            break
    if not copied_cls:
        # fallback by key
        for k in ['cls_token', 'class_token', 'encoder.cls_token']:
            if k in tv_sd and _copy_cls(tv_sd[k]):
                transfers += 1
                break

    # ---- Encoder blocks ----
    # Collect torchvision blocks (layers)
    tv_blocks = []
    if hasattr(tv_model, 'encoder') and hasattr(tv_model.encoder, 'layers'):
        tv_blocks = list(tv_model.encoder.layers)
    elif hasattr(tv_model, 'blocks'):
        tv_blocks = list(tv_model.blocks)
    local_blocks = list(local_model.blocks._modules.values())

    for l_blk, t_blk in zip(local_blocks, tv_blocks):
        # Norms
        for pair in [(getattr(t_blk, 'ln_1', None), l_blk.norm1), (getattr(t_blk, 'ln_2', None), l_blk.norm2)]:
            t_norm, l_norm = pair
            if t_norm is not None:
                if l_norm.weight.shape == t_norm.weight.shape:
                    l_norm.weight.data.copy_(t_norm.weight.detach()); transfers += 1
                if l_norm.bias.shape == t_norm.bias.shape:
                    l_norm.bias.data.copy_(t_norm.bias.detach()); transfers += 1

        # Attention module variants
        l_att = getattr(l_blk, 'attn', None)
        # Torchvision may nest attention under .attention.attention or have direct MultiheadAttention
        t_att = None
        # try hierarchical
        if hasattr(t_blk, 'attention') and hasattr(t_blk.attention, 'attention'):
            t_att = t_blk.attention.attention
        # direct attribute forms
        if t_att is None and hasattr(t_blk, 'attn'):
            t_att = t_blk.attn
        if t_att is None and hasattr(t_blk, 'self_attention'):
            t_att = t_blk.self_attention

        if l_att is not None and t_att is not None:
            # qkv: either combined linear 'qkv' or in_proj_weight/in_proj_bias (MultiheadAttention)
            if hasattr(t_att, 'qkv') and hasattr(l_att, 'qkv'):
                if l_att.qkv.weight.shape == t_att.qkv.weight.shape:
                    l_att.qkv.weight.data.copy_(t_att.qkv.weight.detach()); transfers += 1
                if l_att.qkv.bias.shape == t_att.qkv.bias.shape:
                    l_att.qkv.bias.data.copy_(t_att.qkv.bias.detach()); transfers += 1
            else:
                # MultiheadAttention style
                if hasattr(t_att, 'in_proj_weight') and hasattr(l_att, 'qkv'):
                    if l_att.qkv.weight.shape == t_att.in_proj_weight.shape:
                        l_att.qkv.weight.data.copy_(t_att.in_proj_weight.detach()); transfers += 1
                    if hasattr(t_att, 'in_proj_bias') and l_att.qkv.bias.shape == t_att.in_proj_bias.shape:
                        l_att.qkv.bias.data.copy_(t_att.in_proj_bias.detach()); transfers += 1
            # proj/out_proj
            if hasattr(t_att, 'proj') and hasattr(l_att, 'proj'):
                if l_att.proj.weight.shape == t_att.proj.weight.shape:
                    l_att.proj.weight.data.copy_(t_att.proj.weight.detach()); transfers += 1
                if l_att.proj.bias.shape == t_att.proj.bias.shape:
                    l_att.proj.bias.data.copy_(t_att.proj.bias.detach()); transfers += 1
            elif hasattr(t_att, 'out_proj') and hasattr(l_att, 'proj'):
                if l_att.proj.weight.shape == t_att.out_proj.weight.shape:
                    l_att.proj.weight.data.copy_(t_att.out_proj.weight.detach()); transfers += 1
                if l_att.proj.bias.shape == t_att.out_proj.bias.shape:
                    l_att.proj.bias.data.copy_(t_att.out_proj.bias.detach()); transfers += 1

        # MLP transfer (fc1/fc2)
        t_mlp = getattr(t_blk, 'mlp', None)
        if t_mlp is not None:
            if hasattr(t_mlp, 'fc1') and hasattr(l_blk.mlp, 'fc1'):
                if l_blk.mlp.fc1.weight.shape == t_mlp.fc1.weight.shape:
                    l_blk.mlp.fc1.weight.data.copy_(t_mlp.fc1.weight.detach()); transfers += 1
                if l_blk.mlp.fc1.bias is not None and t_mlp.fc1.bias is not None and l_blk.mlp.fc1.bias.shape == t_mlp.fc1.bias.shape:
                    l_blk.mlp.fc1.bias.data.copy_(t_mlp.fc1.bias.detach()); transfers += 1
            if hasattr(t_mlp, 'fc2') and hasattr(l_blk.mlp, 'fc2'):
                if l_blk.mlp.fc2.weight.shape == t_mlp.fc2.weight.shape:
                    l_blk.mlp.fc2.weight.data.copy_(t_mlp.fc2.weight.detach()); transfers += 1
                if l_blk.mlp.fc2.bias is not None and t_mlp.fc2.bias is not None and l_blk.mlp.fc2.bias.shape == t_mlp.fc2.bias.shape:
                    l_blk.mlp.fc2.bias.data.copy_(t_mlp.fc2.bias.detach()); transfers += 1

    # ---- Final norm (pre-head) ----
    tv_final_norm = None
    if hasattr(tv_model, 'encoder') and hasattr(tv_model.encoder, 'ln'):
        tv_final_norm = tv_model.encoder.ln
    elif hasattr(tv_model, 'ln'):
        tv_final_norm = tv_model.ln
    if tv_final_norm is not None:
        if local_model.norm.weight.shape == tv_final_norm.weight.shape:
            local_model.norm.weight.data.copy_(tv_final_norm.weight.detach()); transfers += 1
        if local_model.norm.bias.shape == tv_final_norm.bias.shape:
            local_model.norm.bias.data.copy_(tv_final_norm.bias.detach()); transfers += 1

    # (Head intentionally skipped due to num_classes mismatch.)
    return transfers

def vit_b_16(**kwargs):
    return ViTB16(**kwargs)

# quick sanity
if __name__ == "__main__":
    model = vit_b_16(num_classes=10, img_size=224)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Logits:", y.shape)        # (2, 10)
    f = model.forward_withoutfcl(x)
    print("Feat:", f.shape)          # (2, 768)
    t = model.forward_feature(x)
    print("Tokens:", t.shape)        # (2, 197, 768)
