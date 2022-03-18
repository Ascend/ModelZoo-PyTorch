#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.


import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.efficientnet_blocks import SqueezeExcite
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

__all__ = [
    'S60','S120',
    'B60','B120',
    'L60','L120'
]

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
    

    
class Learned_Aggregation_Layer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        
        return x_cls
    
        
class Layer_scale_init_Block_only_token(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Learned_Aggregation_Layer, Mlp_block=Mlp,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x, x_cls):
        u = torch.cat((x_cls,x),dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Conv_blocks_se(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.qkv_pos = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, groups = dim, kernel_size = 3, padding = 1, stride = 1, bias = True),
            nn.GELU(),
            SqueezeExcite(dim, rd_ratio=0.25),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.transpose(-1,-2)
        x = x.reshape(B,C,H,W)
        x = self.qkv_pos(x)
        x = x.reshape(B,C,N)
        x = x.transpose(-1,-2)
        return x

    
class Layer_scale_init_Block(nn.Module):

    def __init__(self, dim,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = None,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
    )

class ConvStem(nn.Module):
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

        self.proj = torch.nn.Sequential(
            conv3x3(3, embed_dim // 8, 2),
            nn.GELU(),
            conv3x3(embed_dim // 8, embed_dim // 4, 2),
            nn.GELU(),
            conv3x3(embed_dim // 4, embed_dim // 2, 2),
            nn.GELU(),
            conv3x3(embed_dim // 2, embed_dim, 2),
        )

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
              
class PatchConvnet(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=1, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Layer_scale_init_Block,
                 block_layers_token = Layer_scale_init_Block_only_token,
                 Patch_layer=ConvStem,act_layer=nn.GELU,
                 Attention_block = Conv_blocks_se ,
                dpr_constant=True,init_scale=1e-4,
                Attention_block_token_only=Learned_Aggregation_Layer,
                Mlp_block_token_only= Mlp,
                depth_token_only=1,
                mlp_ratio_clstk = 3.0):
        super().__init__()

        
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))

        if not dpr_constant:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate for i in range(depth)]
            
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,init_values=init_scale)
            for i in range(depth)])
                    
        
        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=int(embed_dim), num_heads=num_heads, mlp_ratio=mlp_ratio_clstk,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
        
        self.norm = norm_layer(int(embed_dim))
        
        self.total_len = depth_token_only+depth
        
        self.feature_info = [dict(num_chs=int(embed_dim ), reduction=0, module='head')]
        self.head = nn.Linear(int(embed_dim), num_classes) if num_classes > 0 else nn.Identity()

        self.rescale = .02

        trunc_normal_(self.cls_token, std=self.rescale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.rescale)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        for i , blk in enumerate(self.blocks):
            x  = blk(x)

        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x  = self.forward_features(x)
        x = self.head(x)
        return x
        
@register_model
def S60(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=384, depth=60, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,**kwargs)

    return model
    
@register_model
def S120(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=384, depth=120, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,**kwargs)

    return model
    
@register_model
def B60(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=768, depth=60, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Attention_block = Conv_blocks_se,
        init_scale=1e-6,**kwargs)

    return model


@register_model
def B120(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=768, depth=120, num_heads=1,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        init_scale=1e-6,**kwargs)

    return model


@register_model
def L60(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=1024, depth=60, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,**kwargs)

    return model



@register_model
def L120(pretrained=False, **kwargs):
    model = PatchConvnet(
        patch_size=16, embed_dim=1024, depth=120, num_heads=1, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block = Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,**kwargs)

    return model
