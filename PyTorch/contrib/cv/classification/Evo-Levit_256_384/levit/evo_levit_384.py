# encoding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import utils
import torch.nn as nn

from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model

specification = {
    'EvoLeViT_128S_384': {
        'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth'},
    'EvoLeViT_128_384': {
        'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth'},
    'EvoLeViT_192_384': {
        'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth'},
    'EvoLeViT_256_384': {
        'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth'},
    'EvoLeViT_384_384': {
        'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1,
        'weights': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth'},
}

prune_ratio_list = {
    'EvoLeViT_128S_384': [[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'EvoLeViT_128_384': [[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]],
    'EvoLeViT_192_384': [[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]],
    'EvoLeViT_256_384': [[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]],
    'EvoLeViT_384_384': [[1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]],
}

__all__ = [specification.keys()]


@register_model
def EvoLeViT_128S_384(num_classes=1000, distillation=True,
                      pretrained=False, fuse=False):
    return model_factory(**specification['EvoLeViT_128S_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         prune_ratio=prune_ratio_list['EvoLeViT_128S_384'])


@register_model
def EvoLeViT_128_384(num_classes=1000, distillation=True,
                     pretrained=False, fuse=False):
    return model_factory(**specification['EvoLeViT_128_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         prune_ratio=prune_ratio_list['EvoLeViT_128_384'])


@register_model
def EvoLeViT_192_384(num_classes=1000, distillation=True,
                     pretrained=False, fuse=False):
    return model_factory(**specification['EvoLeViT_192_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         prune_ratio=prune_ratio_list['EvoLeViT_192_384'])


@register_model
def EvoLeViT_256_384(num_classes=1000, distillation=True,
                     pretrained=False, fuse=False):
    return model_factory(**specification['EvoLeViT_256_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         prune_ratio=prune_ratio_list['EvoLeViT_256_384'])


@register_model
def EvoLeViT_384_384(num_classes=1000, distillation=True,
                     pretrained=False, fuse=False):
    return model_factory(**specification['EvoLeViT_384_384'], num_classes=num_classes,
                         distillation=distillation, pretrained=pretrained, fuse=fuse,
                         prune_ratio=prune_ratio_list['EvoLeViT_384_384'])


global_attn = 0
ori_indices = None
learn_tradeoff_mode = True


def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out


def merge_tokens(x_drop, score):
    # score B,N
    # scale
    weight = score / torch.sum(score, dim=1, keepdim=True)
    x_drop = weight.unsqueeze(-1) * x_drop
    return torch.sum(x_drop, dim=1, keepdim=True)


class CatModule(torch.nn.Module):
    def __init__(self, m1, m2, prune_ratio, N):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.prune_ratio = prune_ratio
        # self.i = i
        if prune_ratio < 1.0:
            N_ = N - int(N * prune_ratio)
            self.drop_fc = nn.AdaptiveAvgPool1d(1)
            # self.recover_fc=nn.Linear(1,N_)

    def set_prune_ratio(self, prune_ratio):
        self.prune_ratio = prune_ratio

    def forward(self, x_):
        global global_attn  # ga
        global ori_indices  # oi
        if self.prune_ratio < 1:
            x = x_[:, 1:]  # split out cls token

            N = x.shape[1]
            N_ = int(N * self.prune_ratio)
            global_attn = global_attn.clone()
            indices = torch.argsort(global_attn, dim=1, descending=True)

            x_ga_oi = torch.cat((x, global_attn.unsqueeze(-1), ori_indices.unsqueeze(-1).half()), dim=-1)
            x_ga_oi = easy_gather(x_ga_oi, indices)
            x_ga_oi = x_ga_oi.contiguous()  # todo
            x_sorted, global_attn, ori_indices = x_ga_oi[:, :, :-2].clone(), x_ga_oi[:, :, -2].clone(), x_ga_oi[:, :, -1].clone()  # todo

            if self.training:
                x_ = torch.cat((x_[:, :1], x_sorted), dim=1)
            else:
                x_[:, 1:] = x_sorted
            x = x_[:, :N_ + 1]
            x_drop = x_[:, N_ + 1:]

            add_token = merge_tokens(x_drop, global_attn[:, N_:])  # B,1,C
            x = torch.cat((x, add_token), dim=1)  # B,N+1,C

            x, raw_x1 = self.m1(x)
            x, raw_x2 = self.m2(x)
            x = x[:, :-1]

            # fast update via skip connection
            add_token1 = raw_x1[:, -1:]
            add_token2 = raw_x2[:, -1:]
            x_drop = x_drop + add_token1.expand(-1, x_drop.shape[1], -1) + add_token2.expand(-1, x_drop.shape[1], -1)

            x_ = torch.cat((x, x_drop), dim=1)
            # x_[:, N_ + 1:] = x_drop
            # x_[:, :N_ + 1] = x
        else:
            x_, _ = self.m1(x_)
            x_, _ = self.m2(x_)
        return x_


class StageModule(torch.nn.Module):
    def __init__(self, m, prune_ratio):
        super().__init__()
        self.m = m
        self.prune_ratio = prune_ratio

    def forward(self, x_):
        global global_attn  # ga
        global ori_indices  # oi

        if isinstance(x_, tuple):
            x_ = x_[0]

        if self.prune_ratio < 1:
            x = x_[:, 1:]  # split out cls token

            N = x.shape[1]
            N_ = int(N * self.prune_ratio)
            indices = torch.argsort(global_attn, dim=1, descending=True)
            x_ga_oi = torch.cat((x, global_attn.unsqueeze(-1), ori_indices.unsqueeze(-1).half()), dim=-1)
            x_ga_oi = easy_gather(x_ga_oi, indices)
            x_ga_oi = x_ga_oi.contiguous()  # todo
            x_sorted, global_attn, ori_indices = x_ga_oi[:, :, :-2].clone(), x_ga_oi[:, :, -2].clone(), x_ga_oi[:, :, -1].clone()  # todo

            if self.training:
                x_ = torch.cat((x_[:, :1], x_sorted), dim=1)
            else:
                x_[:, 1:] = x_sorted

            x = x_[:, :N_ + 1].clone()  # todo
            x_drop = x_[:, N_ + 1:].clone()  # todo

            merge_weight = global_attn[:, N_:].clone()  # todo
            add_token = merge_tokens(x_drop, merge_weight)  # B,1,C
            x = torch.cat((x, add_token), dim=1)  # B,N+1,C

            raw_total = 0
            for blk in self.m:
                x, raw = blk(x)
                raw_total = raw_total + raw[:, -1:].clone()  # todo

            x_drop = x_drop + raw_total.expand(-1, x_drop.shape[1], -1)

            x = x[:, :-1]
            if self.training:
                x_ = torch.cat((x, x_drop), dim=1)
            else:
                x_[:, N_ + 1:] = x_drop
                x_[:, :N_ + 1] = x
        else:
            x_ = self.m(x_)
        return x_


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1), w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation, resolution=224):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(torch.nn.Module):
    def __init__(self, m, drop, out_raw=False):
        super().__init__()
        self.m = m
        self.drop = drop
        self.out_raw = out_raw

    def set_prune_ratio(self, prune_ratio):
        pass

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if self.training and self.drop > 0:
            raw = self.m(x) * torch.rand(x.size(0), 1, 1,
                                         device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            raw = self.m(x)
        if self.out_raw:
            return x + raw, raw
        else:
            return x + raw


class  MatmulApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, mat2):
        ctx.save_for_backward(self, mat2)
        result = torch.matmul(self, mat2.transpose(-2, -1))
        return result
    @staticmethod
    def backward(ctx, grad):
        self, mat2 = ctx.saved_tensors
        self_grad = torch_npu.npu_bmmV2(grad, mat2, [])
        mat2_grad = torch_npu.npu_bmmV2(grad.transpose(-2, -1), self, [])
        return self_grad, mat2_grad

matmul_transpose = MatmulApply.apply


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14, posembed=False, global_attn_tradeoff=0.5):
        super().__init__()
        self.tradeoff = global_attn_tradeoff

        self.learn_tradeoff = torch.nn.Parameter(torch.Tensor([0]))
        self.sigmoid = torch.nn.Sigmoid()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, dim, bn_weight_init=0, resolution=resolution))

        self.pos_embed = posembed

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab

    def forward(self, x):  # x (B,N,C)
        global global_attn
        global learn_tradeoff_mode

        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # attn_raw = (q @ k.transpose(-2, -1)) * self.scale
        attn_raw = matmul_transpose(q, k) * self.scale  # todo

        attn = attn_raw.softmax(dim=-1)

        # update global attn
        if learn_tradeoff_mode:
            tradeoff = self.sigmoid(self.learn_tradeoff)
        else:
            tradeoff = self.tradeoff

        if isinstance(global_attn, int):
            cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)  # B,N
            global_attn = cls_attn
        else:
            if global_attn.shape[1] - N + 2 == 1:
                # no additional token and no pruning
                cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
                global_attn = (1 - tradeoff) * global_attn + tradeoff * cls_attn
            else:
                cls_attn = torch.mean(attn[:, :, 0, 1:-1], dim=1)

                if self.training:
                    temp_attn = (1 - tradeoff) * global_attn[:, :N - 2] + tradeoff * cls_attn
                    global_attn = torch.cat((temp_attn, global_attn[:, N - 2:]), dim=1)
                else:
                    global_attn[:, :N - 2] = (1 - tradeoff) * global_attn[:, :N - 2] + tradeoff * cls_attn

        # x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = torch_npu.npu_confusion_transpose((attn @ v), [0, 2, 1, 3], (B, N, self.dh), True)  # todo
        x = self.proj(x)
        return x


class Subsample(torch.nn.Module):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x, with_cls=True):
        if with_cls:
            B, N, C = x.shape
            x1 = x[:, 1:, :]
            x1 = x1.view(B, self.resolution, self.resolution, C)[
                 :, ::self.stride, ::self.stride].reshape(B, -1, C).contiguous()
            x = torch.cat((x[:, :1, :], x1), dim=1)
        else:
            B, N, C = x.shape
            x = x.view(B, self.resolution, self.resolution, C)[
                :, ::self.stride, ::self.stride].reshape(B, -1, C).contiguous()
        return x


class AttentionSubsample(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7, posembed=False, global_attn_tradeoff=0.5):
        super().__init__()
        self.tradeoff = global_attn_tradeoff

        self.learn_tradeoff = torch.nn.Parameter(torch.Tensor([0]))
        self.sigmoid = torch.nn.Sigmoid()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h, resolution=resolution)

        self.q = torch.nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.pos_embed = posembed
        if posembed:
            self.poss = nn.Parameter(torch.zeros(1, resolution ** 2 + 1, in_dim))
            trunc_normal_(self.poss, std=.02)

        self.stride = stride
        self.resolution = resolution

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab

    def set_prune_ratio(self, prune_ratio):
        pass

    def forward(self, x):
        global global_attn  # ga
        global ori_indices  # oi
        global learn_tradeoff_mode

        if isinstance(x, tuple):
            x = x[0]

        # recover sequence
        old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)

        x_patch = x[:, 1:]
        ori_indices = ori_indices.clone()
        indices = torch.argsort(ori_indices, dim=1)
        x_ga_oi = torch.cat((x_patch, global_attn.unsqueeze(-1), ori_indices.unsqueeze(-1)), dim=-1)
        x_ga_oi = easy_gather(x_ga_oi, indices)
        x_ga_oi = x_ga_oi.contiguous()  # todo
        x_patch, ga_oi = x_ga_oi[:, :, :-2].clone(), x_ga_oi[:, :, -2:].clone()  # todo

        # subsample global attn and ori indices
        ga_oi = self.q[0](ga_oi, False)
        global_attn, ori_indices = ga_oi[:, :, 0].clone(), ga_oi[:, :, 1].clone()  # todo

        # global_attn, ori_indices = ga_oi[:, :, 0], ga_oi[:, :, 1]

        if self.training:
            x = torch.cat((x[:, :1], x_patch), dim=1)
        else:
            x[:, 1:] = x_patch

        x = x + self.poss
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3).contiguous()    # BHNC
        v = v.permute(0, 2, 1, 3).contiguous()    # BHNC
        q = self.q(x).view(B, self.resolution_2 + 1, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        # attn_raw = (q @ k.transpose(-2, -1)) * self.scale
        attn_raw = matmul_transpose(q, k) * self.scale  # todo

        attn = attn_raw.softmax(dim=-1)

        cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)  # B,N
        cls_attn = self.q[0](cls_attn.unsqueeze(-1), False).squeeze(-1)

        if learn_tradeoff_mode:
            tradeoff = self.sigmoid(self.learn_tradeoff)
        else:
            tradeoff = self.tradeoff

        global_attn = (1 - tradeoff) * global_attn + tradeoff * cls_attn

        # normalize global attention
        new_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
        scale = old_global_scale / new_global_scale
        global_attn = global_attn * scale

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(torch.nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=384,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 distillation=True,
                 drop_path=0, prune_ratio=None):
        super().__init__()

        self.stage_wise_prune = True

        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[0]))

        self.patch_embed = hybrid_backbone

        self.pos_embed = True

        self.blocks = []
        self.stage_blocks = []

        down_ops.append([''])
        resolution = img_size // patch_size
        if self.pos_embed:
            self.poss = nn.Parameter(torch.zeros(1, resolution ** 2 + 1, embed_dim[0]))
            trunc_normal_(self.poss, std=.02)

        self.prune_ratio = prune_ratio[0]
        self.stage_prune_ratio = prune_ratio[1]

        layer_index = -1
        n = 14
        j = 0

        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            stage_subblocks = []
            for _ in range(dpth):
                layer_index += 1

                m1 = Residual(Attention(
                    ed, kd, nh,
                    attn_ratio=ar,
                    activation=attention_activation,
                    resolution=resolution,
                    posembed=self.pos_embed
                ), drop_path, out_raw=True)
                if self.prune_ratio[layer_index] == 1:
                    self.stage_blocks.append(m1)
                else:
                    stage_subblocks.append(m1)

                if mr > 0:
                    h = int(ed * mr)
                    m2 = Residual(torch.nn.Sequential(
                        Linear_BN(ed, h, resolution=resolution),
                        mlp_activation(),
                        Linear_BN(h, ed, bn_weight_init=0,
                                  resolution=resolution),
                    ), drop_path, out_raw=True)
                else:
                    m2 = torch.nn.Identity()

                if self.prune_ratio[layer_index] == 1:
                    self.stage_blocks.append(m2)
                else:
                    stage_subblocks.append(m2)

                self.blocks.append(CatModule(m1, m2, prune_ratio=self.prune_ratio[layer_index], N=n ** 2))
                if self.prune_ratio[layer_index] < 1:
                    j = j + 1

            if len(stage_subblocks) != 0:
                stage_subblocks = torch.nn.ModuleList(stage_subblocks)
                self.stage_blocks.append(StageModule(stage_subblocks, prune_ratio=self.stage_prune_ratio[i]))

            if do[0] == 'Subsample':
                n = int((n + 1) / 2)
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                subsample = AttentionSubsample(
                    *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                    attn_ratio=do[3],
                    activation=attention_activation,
                    stride=do[5],
                    resolution=resolution,
                    resolution_=resolution_,
                    posembed=self.pos_embed)
                self.blocks.append(subsample)
                self.stage_blocks.append(subsample)

                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    ffn = Residual(torch.nn.Sequential(
                        Linear_BN(embed_dim[i + 1], h,
                                  resolution=resolution),
                        mlp_activation(),
                        Linear_BN(
                            h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                    ), drop_path)
                    self.blocks.append(ffn)
                    self.stage_blocks.append(ffn)

        self.blocks = torch.nn.Sequential(*self.blocks)
        self.stage_blocks = torch.nn.Sequential(*self.stage_blocks)

        # Classifier head
        self.head = BN_Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.clsc = True
        if self.clsc:
            self.head_cls = BN_Linear(
                embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
            if distillation:
                self.head_cls_dist = BN_Linear(
                    embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'poss' in x}

    def set_learn_tradeoff(self, mode):
        global learn_tradeoff_mode
        learn_tradeoff_mode = mode

    def set_prune_ratio(self, mode):
        pass

    def remove_cls(self):
        if hasattr(self, 'head_cls'):
            del self.head_cls
        if hasattr(self, 'head_cls_dist'):
            del self.head_cls_dist

    def forward(self, x):
        global global_attn
        global ori_indices
        global learn_tradeoff_mode

        global_attn = 0

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        ori_indices = torch.arange(x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)
        ori_indices = ori_indices.expand(x.shape[0], -1)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), 1)
        if self.pos_embed:
            x = x + self.poss

        if self.stage_wise_prune:
            x = self.stage_blocks(x)
        else:
            x = self.blocks(x)

        cls = x[:, 0, :]
        x = x[:, 1:, :]
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if self.clsc:
                if self.training:
                    xcls = self.head_cls(cls)
                    xcls_dist = self.head_cls_dist(cls)
                    return x[0], x[1], xcls, xcls_dist
                else:
                    return (x[0] + x[1]) / 2
            if not self.training:
                x = (x[0] + x[1]) / 2

        else:
            x = self.head(x)
        return x


def model_factory(C, D, X, N, drop_path, weights,
                  num_classes, distillation, pretrained, fuse, prune_ratio):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        num_classes=num_classes,
        drop_path=drop_path,
        distillation=distillation,
        prune_ratio=prune_ratio
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    if fuse:
        utils.replace_batchnorm(model)

    return model


if __name__ == '__main__':
    if __name__ == '__main__':
        for name in specification:
            net = globals()[name](fuse=False, pretrained=False)
            net.eval()
            net.remove_cls()
            net(torch.randn(2, 3, 384, 384))
            print(name, 'Parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))
