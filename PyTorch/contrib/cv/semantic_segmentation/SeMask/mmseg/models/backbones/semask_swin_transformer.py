# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from apex import amp

_LAYERNORM_FORMAT_NZ = True
_LAYERNORM_FORMAT_NZ_BLACKLIST = {192, 384, 768, 1536}

class FastGELU(nn.Module):
    """fast version of nn.GELU()"""

    @staticmethod
    def forward(x):
        return torch_npu.fast_gelu(x)

def npu_drop_path(x, random_tensor, keep_prob: float = 0.):
    """
    Less ops than timm version.
    Async generating and applying of random tensor for accelerating.
    """
    random_tensor += keep_prob
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPathTask:
    def __init__(self, shape, device, dtype, ndim, drop_prob):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.ndim = ndim
        self.drop_prob = drop_prob

        self.request_count = 0
        self.rand_queue = []


class NpuDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks.)
    """
    task_dict = {}
    droppath_stream = None

    def __init__(self, drop_prob=None):
        super(NpuDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = x.dtype
            device = x.device
            ndim = x.ndim
        else:
            raise RuntimeError("input type error!")

        key = (shape, device, dtype, ndim)
        if key not in NpuDropPath.task_dict:
            droppath_task = DropPathTask(shape, device, dtype, ndim, self.drop_prob)
            droppath_task.request_count += 1
            NpuDropPath.task_dict[key] = droppath_task
        elif not NpuDropPath.task_dict[key].rand_queue:
            NpuDropPath.task_dict[key].request_count += 1
        else:
            random_tensor = NpuDropPath.task_dict[key].rand_queue.pop(0)
            return npu_drop_path(x, random_tensor, self.keep_prob)

        return x

    @classmethod
    def enable_droppath_ensemble(cls, model):
        if cls.droppath_stream is None:
            cls.droppath_stream = torch.npu.Stream()

        def wait_stream_hook():
            def hook_function(module, inputs):
                torch.npu.current_stream().wait_stream(cls.droppath_stream)
            return hook_function
        model.register_forward_pre_hook(wait_stream_hook())

        def random_tensor_gen_hook():
            def hook_function(module, inputs):
                with torch.npu.stream(cls.droppath_stream):
                    with torch.no_grad():
                        for _, task in cls.task_dict.items():
                            if len(task.rand_queue) >= task.request_count:
                                continue
                            for i in range(task.request_count - len(task.rand_queue)):
                                shape = (task.shape[0],) + (1,) * (task.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
                                random_tensor = torch.rand(shape, dtype=task.dtype, device=task.device)
                                task.rand_queue.append(random_tensor)
            return hook_function
        model.register_forward_pre_hook(random_tensor_gen_hook())

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=FastGELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MatmulApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, mat2):
        # y = a * b^T
        ctx.save_for_backward(self, mat2)
        result = torch.matmul(self, mat2.transpose(-2, -1))
        return result
    @staticmethod
    def backward(ctx, grad):
        # da: grad * b
        # db: grad^T * a
        self, mat2 = ctx.saved_tensors
        self_grad = torch_npu.npu_bmmV2(grad, mat2, [])
        mat2_grad = torch_npu.npu_bmmV2(grad.transpose(-2, -1), self, [])
        return self_grad, mat2_grad

matmul_transpose = MatmulApply.apply

class RollIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index_fp, index_bp):
        N, H, W, C = input.shape
        ctx.input = input
        ctx.index_bp = index_bp
        result = input.reshape(N, H * W, C).index_select(1, index_fp).reshape(N, H, W, C)
        return result
    @staticmethod
    def backward(ctx, grad):
        input = ctx.input
        N, H, W, C = input.shape
        index_bp = ctx.index_bp
        grad_input = grad.reshape(N, H * W, C).index_select(1, index_bp).reshape(N, H, W, C)
        return grad_input, None, None

roll_index_select = RollIndexSelect.apply

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C) #.contiguous()
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B_, H_, W_, C_ = windows.shape

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)

    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    C = int((B_ * H_ * W_ * C_) / (B * H * W))
    x = torch_npu.npu_confusion_transpose(x, [0, 1, 3, 2, 4, 5], (B, H, W, C), True)

    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = torch.tensor(qk_scale) if qk_scale else torch.tensor(head_dim ** -0.5)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0) #.contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    @amp.half_function
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #.contiguous().npu_format_cast(2)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if not self.scale.device == q.device:
            self.scale = self.scale.to(q.device).to(q.dtype)

        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        attn = matmul_transpose(q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0).half()

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        #attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = torch_npu.npu_confusion_transpose(torch_npu.npu_format_cast((attn @ v), 2), [0, 2, 1, 3], (B_, N, C), True)
        x = self.proj(x)
        #x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def get_roll_index(H, W, shifts):
    index = torch.arange(0, H * W).reshape(H, W)
    index_fp = torch.roll(index, shifts=(shifts, shifts), dims=(0, 1)).reshape(-1).long()
    index_bp = {i:idx for idx, i in enumerate(index_fp.numpy().tolist())}
    index_bp = [index_bp[i] for i in range(H * W)]
    index_bp = torch.LongTensor(index_bp)
    return [index_fp, index_bp]
    

class NpuSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, H, W):
        B, Hp, Wp, C = input.shape
        ctx.input = input
        result = torch_npu.npu_indexing(input, [0, 0, 0, 0], [B, H, W, C], [1, 1, 1, 1])
        return result
    @staticmethod
    def backward(ctx, grad):
        B, H, W, C = grad.shape
        input = ctx.input
        _, Hp, Wp, _ = input.shape
        pads = (0, 0, 0, Hp - H, 0, Wp - W, 0, 0)
        self_grad = torch_npu.npu_pad(grad, pads)
        return self_grad, None, None

npu_slice = NpuSlice.apply

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=FastGELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = NpuDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None   
        self.index_dict = None 
    
    def cast_index_device(self, device):
        for v in self.index_dict.values():
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
         
    def rid(self, device):             
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            Hp = int(np.ceil(self.H / self.window_size)) * self.window_size
            Wp = int(np.ceil(self.W / self.window_size)) * self.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(device)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        if self.shift_size > 0:
            self.index_dict={}
            self.index_dict[(Hp, Wp, self.shift_size)] = get_roll_index(Hp, Wp, self.shift_size)
            self.index_dict[(Hp, Wp, -self.shift_size)] = get_roll_index(Hp, Wp, -self.shift_size)   
            self.cast_index_device(device)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            if self.index_dict is None or (Hp, Wp, -self.shift_size) not in self.index_dict:
                self.rid(x.device)
            index_fp = self.index_dict[(Hp, Wp, -self.shift_size)][0]
            index_bp = self.index_dict[(Hp, Wp, -self.shift_size)][1]
            shifted_x = roll_index_select(x, index_fp, index_bp)
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            index_fp = self.index_dict[(Hp, Wp, self.shift_size)][0]
            index_bp = self.index_dict[(Hp, Wp, self.shift_size)][1]
            x = roll_index_select(shifted_x, index_fp, index_bp)
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = npu_slice(x, H, W) #x[:, :H, :W, :] #.contiguous()

        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        if _LAYERNORM_FORMAT_NZ and x.size(-1) not in _LAYERNORM_FORMAT_NZ_BLACKLIST:
            x = x + self.drop_path(self.mlp(self.norm2(torch_npu.npu_format_cast(x, 29))))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SemanticAttention(nn.Module):
    """ ClassMasking
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, n_cls):

        super().__init__()
        self.dim = dim
        self.n_cls = n_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.n_cls)
        self.mlp_cls_k =nn.Linear(self.dim, self.n_cls)
        
        self.mlp_v = nn.Linear(self.dim, self.dim)
        
        self.mlp_res = nn.Linear(self.dim, self.dim)
        
        self.proj_drop = nn.Dropout(0.1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.init_weight()
    
    @amp.half_function
    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, N, C)
        returns:
            class_seg_map: (B, N, K)
            gated feats: (B, N, C)
        """

        seg_map = self.mlp_cls_q(x)
        seg_ft = self.mlp_cls_k(x)

        feats = self.mlp_v(x)

        seg_score = matmul_transpose(seg_map, seg_ft)
        seg_score = self.softmax(seg_score)

        feats = seg_score @ feats
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x

        return seg_map, feat_map
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Linear):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.LayerNorm):
                nn.init.constant_(ly.bias, 0)
                nn.init.constant_(ly.weight, 1.0)
        
        nn.init.zeros_(self.mlp_res.weight)
        if not self.mlp_res.bias is None: nn.init.constant_(self.mlp_res.bias, 0)


class SWSeMaskBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, n_cls, window_size=7, num_sem_blocks=1, act_layer=FastGELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_cls = n_cls

        self.norm1 = norm_layer(dim)

        self.class_injection = nn.ModuleList([])

        for i in range(num_sem_blocks):
            self.class_injection.append(SemanticAttention(dim=dim, n_cls=n_cls))
        
        self.H = None
        self.W = None
    
    @amp.half_function
    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        K = self.n_cls

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        for blk in self.class_injection:
            sem_windows, x_windows = blk(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x_windows, self.window_size, Hp, Wp)  # B H' W' C

        # merge windows
        sem_windows = sem_windows.view(-1, self.window_size, self.window_size, K)
        shifted_sem = window_reverse(sem_windows, self.window_size, Hp, Wp)  # B H' W' K

        x = shifted_x
        sem_map = shifted_sem

        if pad_r > 0 or pad_b > 0:
            x = npu_slice(x, H, W)
            sem_map = npu_slice(sem_map, H, W)

        x = x.reshape(B, H * W, C)
        sem_map = sem_map.reshape(B, H * W, K)

        return sem_map, x


class SeMaskBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, n_cls, window_size=7, num_sem_blocks=1, act_layer=FastGELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_cls = n_cls

        self.norm1 = norm_layer(dim)

        self.class_injection = nn.ModuleList([])

        for i in range(num_sem_blocks):
            self.class_injection.append(SemanticAttention(dim=dim, n_cls=n_cls))
        
        self.H = None
        self.W = None
        
    @amp.half_function
    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        K = self.n_cls

        x = self.norm1(x)
        # W-MSA/SW-MSA
        for blk in self.class_injection:
            sem_x, x = blk(x)  # nW*B, window_size*window_size, C

        return sem_x, x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input ffif self.shift_sizeeature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        """
        Using depth-wise conv2d to merge patches for high performance.
        """
        C_list = [96, 192, 384]
        self.kernel_dict = {}
        self.kernel_device = torch.device('cpu')
        for c in C_list:
            kernel0 = torch.FloatTensor([[1, 0], [0, 0]]).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            kernel1 = torch.FloatTensor([[0, 0], [1, 0]]).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            kernel2 = torch.FloatTensor([[0, 1], [0, 0]]).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            kernel3 = torch.FloatTensor([[0, 0], [0, 1]]).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            kernel = torch.cat([kernel0, kernel1, kernel2, kernel3], 0)
            self.kernel_dict[c] = kernel

    def cast_kernel_device(self, device):
        for k, v in self.kernel_dict.items():
            self.kernel_dict[k] = v.to(device)

    @amp.half_function
    def forward(self, x, H, W):
        """
        x: B, H*W, C

        A depth-wise conv2d version with save semantics of following op
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        """

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # depth-conv2d version
        x = x.reshape(B, int(H / 2), 2, int(W / 2), 2, C)
        x = x.permute(0, 1, 3, 4, 2, 5)
        x = x.reshape(B, int(H * W / 4), C * 4)

        if _LAYERNORM_FORMAT_NZ and x.size(-1) not in _LAYERNORM_FORMAT_NZ_BLACKLIST:
            x = torch_npu.npu_format_cast(torch_npu.npu_format_cast(x, 2), 29).contiguous()

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 i,
                 var,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 sem_window_size=7,
                 num_sem_blocks=1,
                 n_cls=150,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_sem_blocks = num_sem_blocks

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        if num_sem_blocks > 0:
            if var is None:
                self.semantic_layer = SWSeMaskBlock(dim=dim, 
                                                n_cls=n_cls, 
                                                num_sem_blocks=num_sem_blocks, 
                                                window_size=sem_window_size)
            elif var == 'large':
                if i in [0,1]:
                    self.semantic_layer = SWSeMaskBlock(dim=dim, 
                                                    n_cls=n_cls, 
                                                    num_sem_blocks=num_sem_blocks, 
                                                    window_size=sem_window_size)
                else:
                    self.semantic_layer = SeMaskBlock(dim=dim, 
                                                    n_cls=n_cls, 
                                                    num_sem_blocks=num_sem_blocks, 
                                                    window_size=sem_window_size)
            

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    @amp.half_function
    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.num_sem_blocks > 0:
            self.semantic_layer.H, self.semantic_layer.W = H, W
            seg_map, x = self.semantic_layer(x)
        else:
            seg_map = None

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, seg_map, H, W, x_down, Wh, Ww
        else:
            return x, seg_map, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
            
    @amp.half_function
    def forward(self, x):
        """Forward function."""
        # padding
        if len(x.shape) == 3:
            x = x.unsqueeze(0) 
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        Wh, Ww = x.size(2), x.size(3)
        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            if _LAYERNORM_FORMAT_NZ and x.size(-1) not in _LAYERNORM_FORMAT_NZ_BLACKLIST:
                x = torch_npu.npu_format_cast(torch_npu.npu_format_cast(x, 2), 29)
            x = self.norm(x)

        return x, Wh, Ww


@BACKBONES.register_module()
class SeMaskSwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 num_cls=150,
                 sem_window_size=7,
                 num_sem_blocks=[1, 1, 1, 1],
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.n_cls = num_cls
        
        if (self.embed_dim == 192 or self.embed_dim == 96) and self.n_cls > 19:
            var = 'large'
        else:
            var = None

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                i=i_layer,
                var=var,
                n_cls=num_cls,
                sem_window_size=sem_window_size,
                num_sem_blocks=num_sem_blocks[i_layer],
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @amp.half_function
    def forward(self, x):
        """Forward function."""
        x, Wh, Ww = self.patch_embed(x)
        outs = []
        cls_outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, cls_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

                cls_out = cls_out.view(-1, H, W, self.n_cls).permute(0, 3, 1, 2).contiguous()
                cls_outs.append(cls_out)

        return tuple(outs), cls_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SeMaskSwinTransformer, self).train(mode)
        self._freeze_stages()
