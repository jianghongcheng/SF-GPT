#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://github.com/caiyuanhao1998/MST-plus-plus
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
from torchvision import models, transforms
from scipy.io import savemat
from torch.nn.init import _calculate_fan_in_and_fan_out


############################# Utility Functions #############################

def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0).detach().numpy()
    print("Feature map shape:", feature_map.shape)
    savemat('test.mat', {"foo": feature_map})
    plt.imshow(feature_map.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


############################# Basic Modules #############################

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride
    )


def shift_back(inputs, step=2):
    bs, nC, row, col = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


############################# Transformer Components #############################

class SFG_MSA(nn.Module):
    def __init__(self, dim_in, dim_head, dim_out, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim_in, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim_in, dim_head * heads, bias=False)

        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim_out, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head, dim_out, 3, 1, 1, bias=False, groups=dim_head),
            GELU(),
            nn.Conv2d(dim_out, dim_out, 3, 1, 1, bias=False, groups=dim_out),
        )

        self.dim = dim_in
        self.dim_out = dim_out

    def forward(self, x_pix, x_dct):
        b, h, w, c = x_pix.shape
        x = x_pix.reshape(b, h * w, c)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (self.to_q(x), self.to_k(x), self.to_v(x))
        )

        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)

        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, self.dim_out)

        out_p = self.pos_emb(v.reshape(b, self.dim_head, h, w)).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SF_GPG(nn.Module):
    def __init__(self, dim_in, dim_head, dim_out, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                SFG_MSA(dim_in=dim_in // 2, dim_head=dim_head, dim_out=dim_out, heads=heads),
                None  # Optionally add: PreNorm(dim_out, FeedForward(dim_out))
            ]) for _ in range(num_blocks)
        ])

    def forward(self, x, x_):
        x, x_ = x.permute(0, 2, 3, 1), x_.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, x_) + x
            # x = ff(x) + x  # Uncomment if using FeedForward
        return x.permute(0, 3, 1, 2)