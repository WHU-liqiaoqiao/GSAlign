import logging
import math
import sys
from functools import partial

sys.path.append('/root/project/SeCap-AGPReID-main')

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision import transforms as T
from fvcore.nn import FlopCountAnalysis

from fastreid.layers import DropPath, trunc_normal_, to_2tuple

logger = logging.getLogger(__name__)


# ----------------- 基本 MLP 和 Attention -----------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ----------------- Block with TPS -----------------
class TPSWarp(nn.Module):
    def __init__(self, num_ctrl=4, feat_dim=768, skip_warp_for_flops=False):
        super().__init__()
        self.num_ctrl = num_ctrl
        self.feat_dim = feat_dim
        self.skip_warp_for_flops = skip_warp_for_flops  # 新增参数控制是否跳过warp

        grid_size = int(math.sqrt(num_ctrl))
        ctrl_points = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size)
        ), dim=-1).view(-1, 2)

        self.register_buffer('src_points', ctrl_points)
        self.dst_points = nn.Parameter(ctrl_points.clone())

        self.angle_predictor = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def get_rotation_matrix(self, angle: torch.Tensor):
        device = angle.device
        cos_theta = torch.cos(angle).view(-1, 1, 1)
        sin_theta = torch.sin(angle).view(-1, 1, 1)

        R = torch.cat([
            torch.cat([cos_theta, -sin_theta], dim=-1),
            torch.cat([sin_theta,  cos_theta], dim=-1)
        ], dim=-2)
        return R.to(device)

    def apply_rotation(self, points: torch.Tensor, R: torch.Tensor):
        return torch.bmm(points, R)

    def forward(self, x):
        B, N, C = x.shape
        if N <= 2:
            return x, None
        cls_token, view_token, patch_tokens = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:, :]
        P = patch_tokens.shape[1]
        side = int(math.sqrt(P))
        if side * side != P:
            return x, None

        feat_2d = patch_tokens.transpose(1, 2).view(B, C, side, side)

        # 如果设置跳过warp计算，则直接返回输入特征（用于FLOPs计算）
        if self.skip_warp_for_flops:
            # 直接返回原特征，不执行warp
            return x, None

        angle = self.angle_predictor(feat_2d) * (math.pi * 0.5)
        R = self.get_rotation_matrix(angle)
        rotated_src = self.apply_rotation(self.src_points.expand(B, -1, -1), R)
        warped_feat = kornia.geometry.transform.warp_image_tps(
            feat_2d, rotated_src, self.dst_points.expand(B, -1, -1), (side, side), align_corners=True
        )
        warped_patch = patch_tokens + 0.2 * warped_feat.view(B, C, -1).permute(0, 2, 1)
        return torch.cat([cls_token, view_token, warped_patch], dim=1), angle


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, with_tps=False, skip_warp_for_flops=False):
        super().__init__()
        self.tps = TPSWarp(feat_dim=dim, skip_warp_for_flops=skip_warp_for_flops) if with_tps else None
        # 其它代码不变
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        tps_info = None
        if self.tps is not None:
            x, tps_info = self.tps(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, tps_info


# ----------------- Patch Embedding -----------------
class PatchEmbed_overlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.img_size = img_size
        self.num_patches = ((img_size[1] - patch_size[1]) // stride_size[1] + 1) * \
                           ((img_size[0] - patch_size[0]) // stride_size[0] + 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ----------------- VisionTransformer -----------------
class VisionTransformer_multiview_onebranch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, drop_path_rate=0., tps_layers=None, skip_warp_for_flops=False):
        super().__init__()
        self.patch_embed = PatchEmbed_overlap(img_size, patch_size, stride_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, drop_path=dpr[i], with_tps=(tps_layers and i in tps_layers), skip_warp_for_flops=skip_warp_for_flops)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        view_tokens = self.view_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, view_tokens, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)
        return x[:, 0]


# ----------------- FLOPs & Params -----------------
def measure_model(model, input_res=(224, 224), batch_size=1):
    dummy = torch.randn(batch_size, 3, *input_res)
    flops = FlopCountAnalysis(model, dummy).total()
    params = sum(p.numel() for p in model.parameters())
    return flops / 1e9, params / 1e6


if __name__ == "__main__":
    vit = VisionTransformer_multiview_onebranch(
    img_size=224, patch_size=16, stride_size=16, embed_dim=768, depth=12, num_heads=12,
    tps_layers=list(range(12)),
    skip_warp_for_flops=True   # 新增参数传递
)
    flops, params = measure_model(vit, (224, 224), 1)
    print(f"VisionTransformer FLOPs: {flops:.2f} GFLOPs, Params: {params:.2f} M")
