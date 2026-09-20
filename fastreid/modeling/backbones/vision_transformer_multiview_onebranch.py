""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
import kornia
from torchvision import transforms as T


logger = logging.getLogger(__name__)


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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
#  TPS 网格生成器（自包含、完全可微，替代 kornia.warp_image_tps）
#  数学正确性已验证：源点==目标点时生成恒等采样网格（误差 ~1e-16）
# ============================================================
def _compute_partial_repr(input_points, control_points):
    """径向基项 U(r)=r^2 * log(r^2)。 input:[N,2] ctrl:[M,2] -> [N,M]"""
    N = input_points.size(0)
    M = control_points.size(0)
    diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    dist = (diff ** 2).sum(-1)                          # 平方距离 [N,M]
    repr_m = 0.5 * dist * torch.log(dist.clamp(min=1e-8))
    return repr_m


class TPSGridGen(nn.Module):
    """给定固定目标控制点，预计算逆核；forward 输入源控制点，输出 [B,HW,2] 采样网格。"""

    def __init__(self, target_h, target_w, target_control_points):
        super().__init__()
        K = target_control_points.size(0)
        self.num_points = K
        self.target_h = target_h
        self.target_w = target_w
        tc = target_control_points.float()

        # 前向核矩阵 [K+3, K+3]
        fk = torch.zeros(K + 3, K + 3)
        fk[:K, :K] = _compute_partial_repr(tc, tc)
        fk[:K, -3] = 1
        fk[-3, :K] = 1
        fk[:K, -2:] = tc
        fk[-2:, :K] = tc.t()
        inverse_kernel = torch.inverse(fk)

        # 输出规则网格坐标 (x, y) ∈ [-1,1]，与 align_corners=True 对齐
        ys, xs = torch.meshgrid(
            torch.arange(target_h), torch.arange(target_w), indexing='ij'
        )
        Y = ys.reshape(-1, 1).float() * 2 / (target_h - 1) - 1
        X = xs.reshape(-1, 1).float() * 2 / (target_w - 1) - 1
        target_coord = torch.cat([X, Y], dim=1)          # [HW,2] 顺序 (x,y)
        target_coord_repr = torch.cat([
            _compute_partial_repr(target_coord, tc),
            torch.ones(target_h * target_w, 1),
            target_coord
        ], dim=1)                                        # [HW, K+3]

        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding', torch.zeros(3, 2))
        self.register_buffer('target_coord_repr', target_coord_repr)

    def forward(self, source_control_points):
        # source_control_points: [B, K, 2]
        B = source_control_points.size(0)
        Y = torch.cat([source_control_points,
                       self.padding.expand(B, 3, 2)], dim=1)   # [B, K+3, 2]
        mapping = torch.matmul(self.inverse_kernel, Y)          # [B, K+3, 2]
        grid = torch.matmul(self.target_coord_repr, mapping)    # [B, HW, 2]
        return grid


# ============================================================
#  LTPS 模块
# ============================================================
class TPSWarp(nn.Module):
    """Learnable Thin Plate Spline (LTPS).

    Args:
        feat_dim: 通道数（ViT-Base=768）
        feat_h, feat_w: patch 特征图高/宽（CARGO=16/8）。必须提供才能正确 reshape。
        num_ctrl: 控制点数，论文最优=4（2x2）
        eta: 残差融合系数 η
    """

    def __init__(self, feat_dim=768, feat_h=None, feat_w=None, num_ctrl=4, eta=0.1):
        super().__init__()
        assert feat_h is not None and feat_w is not None, \
            "TPSWarp 需要真实特征图尺寸 feat_h/feat_w（例如 CARGO 是 16x8）"
        self.feat_dim = feat_dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.num_ctrl = num_ctrl
        self.eta = eta

        grid_size = int(math.sqrt(num_ctrl))
        assert grid_size * grid_size == num_ctrl, "num_ctrl 需为完全平方数(4/9/16/25/36)"
        ctrl_points = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing='ij'
        ), dim=-1).view(-1, 2)  # [num_ctrl, 2]

    
        self.src_points = nn.Parameter(ctrl_points.clone())     # 可学习源点
        self.register_buffer('dst_points', ctrl_points.clone())  # 固定目标点

        # 基于固定目标点预计算 TPS 求解
        self.grid_gen = TPSGridGen(feat_h, feat_w, self.dst_points)

        # 旋转角度预测模块： F -> θ∈[-π/2, π/2]
        self.angle_predictor = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),      # 预测旋转角度
            nn.Tanh()               # 归一到 [-1, 1]
        )

    def forward(self, x):
        # x: [B, N, C], N = 2 + feat_h*feat_w （cls, view, patches）
        B, N, C = x.shape
        num_patch = N - 2
        # 尺寸不符时安全跳过（正常配置下不会触发）
        if num_patch != self.feat_h * self.feat_w:
            return x, None

        cls_token = x[:, 0:1, :]     # [B, 1, C]
        view_token = x[:, 1:2, :]    # [B, 1, C]
        patch_tokens = x[:, 2:, :]   # [B, HW, C]

        # 还原为 2D 特征图 [B, C, H, W]
        feat_2d = patch_tokens.transpose(1, 2).reshape(B, C, self.feat_h, self.feat_w)


        # 预测旋转角 θ∈[-π/2, π/2]
        rotation_angle = self.angle_predictor(feat_2d) * (math.pi * 0.5)  # [B, 1]

        # 旋转源控制点： P_s^rot = P_s · Rᵀ,  Rᵀ=[[cosθ, sinθ], [-sinθ, cosθ]]
        cos_t = torch.cos(rotation_angle)   # [B,1]
        sin_t = torch.sin(rotation_angle)
        Rt = torch.stack([
            torch.cat([cos_t,  sin_t], dim=1),      # 行0: [cos,  sin]
            torch.cat([-sin_t, cos_t], dim=1),      # 行1: [-sin, cos]
        ], dim=1)                                   # [B,2,2] = Rᵀ
        src = self.src_points.unsqueeze(0).expand(B, -1, -1)   # [B,K,2]
        src_rot = torch.bmm(src, Rt)                           # [B,K,2]

        # 生成采样网格（固定目标规则网格 -> 旋转后的源点）并采样
        grid = self.grid_gen(src_rot).view(B, self.feat_h, self.feat_w, 2)
        warped_feat = F.grid_sample(feat_2d, grid, align_corners=True, padding_mode='border')

        # 转回序列
        warped_patch = warped_feat.reshape(B, C, -1).permute(0, 2, 1)   # [B, HW, C]

        # 残差融合 (原始特征 + η*变形特征)
        # eta: CARGO 建议 ~0.2，AG-REID 建议 ~0.5，可在构造时调整
        warped_patch = patch_tokens + self.eta * warped_patch

        # 控制点偏移量（供可视化/记录），rotation_angle 供 deform loss 使用
        delta_control_points = src_rot - self.dst_points.unsqueeze(0)

        out = torch.cat([cls_token, view_token, warped_patch], dim=1)
        return out, (rotation_angle, delta_control_points)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, with_tps=False,
                 feat_h=None, feat_w=None, tps_num_ctrl=4, tps_eta=0.1):
        super().__init__()

        # 在指定层添加 LTPS 模块（需要特征图尺寸）
        self.tps = TPSWarp(feat_dim=dim, feat_h=feat_h, feat_w=feat_w,
                           num_ctrl=tps_num_ctrl, eta=tps_eta) if with_tps else None

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # 在注意力前加入 LTPS（论文：LTPS 插在每个 ViT 层之前）
        tps_info = None
        if self.tps is not None:
            x, tps_info = self.tps(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, tps_info


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        # 记录特征图高/宽，供 LTPS 使用
        self.num_y = feature_size[0]
        self.num_x = feature_size[1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class VisionTransformer_multiview_onebranch(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0, inner_sub=True,
                 tps_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 tps_num_ctrl=4, tps_eta=0.1):
        super().__init__()

        num_classes = 2500

        ######添加losses逻辑
        self.rotation_angle_list = []
        self.delta_list = []

        ######添加一个mask generator模块
        self.mask_generator = ChannelMaskGenerator(embed_dim)

        #####添加结束
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # LTPS 需要的真实特征图尺寸（PatchEmbed_overlap / HybridEmbed 均已提供 num_y/num_x）
        feat_h = getattr(self.patch_embed, 'num_y', int(math.sqrt(num_patches)))
        feat_w = getattr(self.patch_embed, 'num_x', int(math.sqrt(num_patches)))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                with_tps=(i in tps_layers),   # 在指定层启用 LTPS
                feat_h=feat_h, feat_w=feat_w,  # LTPS 需要的真实特征图尺寸
                tps_num_ctrl=tps_num_ctrl,     # 论文最优 K=4
                tps_eta=tps_eta                # 残差融合系数
            ))

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.inner_sub = inner_sub

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_token'}

    def forward(self, x, camera_id=None, labels=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        # 无条件保存 patch tokens（供 TPS 记录用），避免 eval / labels=None 时 NameError
        patch_tokens = x

        ########插入 mask模块
        if self.training and labels is not None:
            # 取 batch 内类中心特征做 mask
            feat = x.mean(dim=1)  # 或使用 cls_token x[:, 0]
            masked_feat, mask_loss = self.mask_generator(feat, labels)
        else:
            mask_loss = None
        ################

        cls_tokens = self.cls_token.expand(B, -1, -1)
        view_tokens = self.view_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, view_tokens, x), dim=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        self.rotation_angle_list.clear()
        self.delta_list.clear()

        tps_features = []

        for i, blk in enumerate(self.blocks):
            x, tps_info = blk(x)
            if tps_info is not None:
                rotation_angle, delta_control_points = tps_info
                self.rotation_angle_list.append(rotation_angle)
                self.delta_list.append(delta_control_points)

                raw_patch = patch_tokens.detach().clone()
                warped_patch = x[:, 2:, :].detach().clone()
                tps_features.append((raw_patch, warped_patch))  # 保存原始和变换后的特征

            if self.inner_sub:
                x[:, 0] = x[:, 0] - x[:, 1]

        x = self.norm(x)

        # 从 transformer 输出中提取 global_feat
        global_feat = x[:, 0]  # (B, D)

        return x[:, 0].reshape(x.shape[0], -1, 1, 1), x[:, 1].reshape(x.shape[0], -1, 1, 1), \
            self.rotation_angle_list, mask_loss, tps_features


class ChannelMaskGenerator(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()  # 输出 channel-wise mask
        )
        self.mse_loss = nn.MSELoss()
        self.entropy_weight = 0.1

    def forward(self, global_feat, labels):
        """
        输入:
            global_feat: [B, D], 图像特征
            labels: [B], 图像标签
        返回:
            mask_loss: 对齐损失 + mask 熵
        """
        B, D = global_feat.shape
        device = global_feat.device

        # === Step 1: 计算当前 batch 内每类的中心特征 ===
        unique_labels = labels.unique()
        centers = torch.zeros_like(global_feat)
        for label in unique_labels:
            idxs = (labels == label)  # 找出该类的索引
            class_feats = global_feat[idxs]  # [N_c, D]
            center = class_feats.mean(dim=0, keepdim=True)  # [1, D]
            centers[idxs] = center  # 分配给该类样本

        centers = F.normalize(centers, dim=-1)
        global_feat = F.normalize(global_feat, dim=-1)

        # === Step 2: 生成 mask ===
        mask = self.mlp(global_feat)  # [B, D]
        masked_proto = centers * mask  # [B, D]
        masked_proto = F.normalize(masked_proto, dim=-1)

        # === Step 3: 构造损失 ===
        align_loss = self.mse_loss(global_feat, masked_proto)  #####CARGO

        # 可选稀疏正则：增强 mask 的选择性
        entropy = -(mask * mask.clamp(min=1e-6).log()).mean()
        mask_loss = align_loss + self.entropy_weight * entropy

        return masked_proto, mask_loss


def resize_pos_embed(posemb, posemb_new, hight, width, cls_token_num):
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :cls_token_num], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


@BACKBONE_REGISTRY.register()
def build_multiview_vit_backbone_onebranch(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off
    input_size = cfg.INPUT.SIZE_TRAIN
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth = cfg.MODEL.BACKBONE.DEPTH
    sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
    stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
    drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    inner_sub = cfg.MODEL.BACKBONE.INNER_SUB
    # fmt: on

    num_depth = {
        'small': 8,
        'base': 12,
    }[depth]

    num_heads = {
        'small': 8,
        'base': 12,
    }[depth]

    mlp_ratio = {
        'small': 3.,
        'base': 4.
    }[depth]

    qkv_bias = {
        'small': False,
        'base': True
    }[depth]

    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
    }[depth]

    model = VisionTransformer_multiview_onebranch(
        img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size,
        depth=num_depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        drop_path_rate=drop_path_ratio, drop_rate=drop_ratio,
        attn_drop_rate=attn_drop_rate, inner_sub=inner_sub,
    )

    if pretrain:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = model.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed(v, model.pos_embed.data, model.patch_embed.num_y, model.patch_embed.num_x, 2)
                state_dict[k] = v
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
    return model