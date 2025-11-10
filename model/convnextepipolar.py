# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn
import timm


logger = logging.getLogger("dinov3")

from model import MODEL

REPO_DIR = "/home/albus/Python_Codes/ADer/model/dinomaly_components/dinov3"
weight_path = "/home/albus/Python_Codes/ADer/ader_weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # 1. Upsample layers
        self.upsample_layers = nn.ModuleList()

        for i in range(3):
            upsample_layer = nn.Sequential(
                LayerNorm(dims[::-1][i], eps=1e-6, data_format="channels_first"),
                nn.ConvTranspose2d(dims[::-1][i], dims[::-1][i + 1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)

        # 2. Stage layers
        self.stages = nn.ModuleList()

        for i in range(3):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[::-1][i+1], drop_path=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[::-1][i+1])
                ]
            )
            self.stages.append(stage)

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        outputs = []
        for i in range(3):
            x = self.upsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs[::-1]

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

#========== MFF & OCE ==========
# class MFF_OCE(nn.Module):
#     def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
#         super(MFF_OCE, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.base_width = width_per_group
#         self.inplanes = 1152
#         self.dilation = 1
#         self.bn_layer = self._make_layer(block, 768, layers, stride=2)

#         self.conv1 = conv3x3(96, 192, 2)
#         self.bn1 = norm_layer(192)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(192, 384, 2)
#         self.bn2 = norm_layer(384)
#         self.conv3 = conv3x3(192, 384, 2)
#         self.bn3 = norm_layer(384)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
    
#     def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation

#         if stride != 1:
#             downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
#                                        norm_layer(planes), )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))

#         for _ in range(1, blocks):
#             layers.append(block(planes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
#         return nn.Sequential(*layers)

#     def _forward_impl(self, x):
#         l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
#         l2 = self.relu(self.bn3(self.conv3(x[1])))
#         feature = torch.cat([l1,l2,x[2]],1)
#         output = self.bn_layer(feature)

#         return output.contiguous()

#     def forward(self, x):
#         return self._forward_impl(x)

# class MFF_OCE_V1(nn.Module):
#     def __init__(self, arch):
#         super(MFF_OCE_V1, self).__init__()

#         dims = convnext_sizes[arch]['dims']

#         self.downsample_layers = nn.ModuleList()

#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 nn.Conv2d(dims[i], dims[-1], kernel_size=2 ** (3-i), stride=2 ** (3-i)),
#                 LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.init_weights()
    
#     def init_weights(self):
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.LayerNorm):
#             module.reset_parameters()
#         if isinstance(module, LayerNorm):
#             module.weight = nn.Parameter(torch.ones(module.normalized_shape))
#             module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
#             torch.nn.init.trunc_normal_(module.weight, std=0.02)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, x_list):
#         out = []
#         for i in range(3):
#             out.append(self.downsample_layers[i](x_list[i]))
#         return torch.stack(out, dim=1).mean(dim=1)

# class MFF_OCE_V2(nn.Module):
#     def __init__(self, arch):
#         super(MFF_OCE_V2, self).__init__()

#         dims = convnext_sizes[arch]['dims']

#         self.downsample_layers = nn.ModuleList()

#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                 nn.Conv2d(dims[i], dims[-1], kernel_size=2 ** (3-i), stride=2 ** (3-i)),
#                 LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.stages = nn.Sequential(
#                 *[
#                     Block(dim=dims[-1], drop_path=0.4, layer_scale_init_value=1e-6)
#                     for j in range(3)
#                 ])

#         self.init_weights()
    
#     def init_weights(self):
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.LayerNorm):
#             module.reset_parameters()
#         if isinstance(module, LayerNorm):
#             module.weight = nn.Parameter(torch.ones(module.normalized_shape))
#             module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
#             torch.nn.init.trunc_normal_(module.weight, std=0.02)
#             nn.init.constant_(module.bias, 0)

#     def forward(self, x_list):
#         out = []
#         for i in range(3):
#             out.append(self.downsample_layers[i](x_list[i]))

#         out = torch.stack(out, dim=1).mean(dim=1)
#         out = self.stages(out)
#         return out

class MFF_OCE_V3(nn.Module):
    def __init__(self, arch):
        super(MFF_OCE_V3, self).__init__()

        dims = convnext_sizes[arch]['dims']

        self.downsample_layers = nn.ModuleList()

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[-1], kernel_size=2 ** (3-i), stride=2 ** (3-i)),
                LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            )
            self.downsample_layers.append(downsample_layer)

        self.reduce_channel_conv = conv1x1(dims[-1]*3, dims[-1])

        self.init_weights()
    
    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x_list):
        out = []
        for i in range(3):
            out.append(self.downsample_layers[i](x_list[i]))
        
        out = torch.cat(out, dim=1)
        out = self.reduce_channel_conv(out)
        return out
        #return torch.stack(out, dim=1).mean(dim=1)

class ConvnextRD(nn.Module):
    def __init__(self):
        super(ConvnextRD, self).__init__()
        #self.net_t = timm.create_model("convnext_tiny.dinov3_lvd1689m", features_only=True, pretrained=True, out_indices=(0,1,2))
        self.net_t = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=weight_path)


        # self.mff_oce = MFF_OCE(block = timm.models.resnet.BasicBlock, layers = 3)
        self.mff_oce = MFF_OCE_V3(arch = "tiny")
        self.net_s = get_convnext_arch("convnext_tiny")()

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs, img_path=None):
        # feats_t = self.net_t(imgs)
        x = imgs
        feats_t = []
        for i in range(3):
            x = self.net_t.downsample_layers[i](x)
            x = self.net_t.stages[i](x)
            feats_t.append(x)


        feats_t = [f.detach() for f in feats_t]
        
        feats_s = self.net_s(self.mff_oce(feats_t))
        return feats_t, feats_s
    
@MODEL.register_module
def convnextrd(pretrained=False, **kwargs):
	model = ConvnextRD(**kwargs)
	return model

# model = get_convnext_arch("convnext_tiny")()
# input = torch.randn(20, 3, 256, 256)
# output = model(input)
# print(model)

# model = timm.create_model("convnext_small.dinov3_lvd1689m", features_only=True, pretrained=True, out_indices=(0,1,2))

# input = torch.randn((20, 3, 256, 256))

# output = model(input)
# for out in output:
#     print(out.shape)

# mff_oce = MFF_OCE(block = timm.models.resnet.BasicBlock, layers = 3)

# output = mff_oce(output)

# student_s = get_convnext_arch("convnext_tiny")()
# output = student_s(output)
# for out in output:
#     print(out.shape)

# model = ConvnextRD()
# input = torch.randn(20, 3, 256, 256)
# feats_t, feats_s = model(input)

