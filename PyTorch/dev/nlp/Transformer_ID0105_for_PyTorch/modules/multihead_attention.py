# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
# -------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


from typing import Dict, Optional
import torch
from torch import nn, Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.variable import Variable
from utils import utils

class QueryLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_q, scale_cpu, scale_npu):

        ctx.save_for_backward(input, weights_q, scale_cpu, scale_npu)
        q = torch.addmm(input.view(input.size(0) * input.size(1), input.size(2)),
                        input.view(input.size(0) * input.size(1), input.size(2)), weights_q, beta=0.0, alpha=scale_cpu)
        q = q.view(input.size(0), input.size(1), input.size(2))
        return q.detach()

    @staticmethod
    def backward(ctx, q_grad):
        input, weights_q, scale_cpu, scale_npu = ctx.saved_tensors
        input = input.view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)
        q = torch.addmm(q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)),
                        q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)), weights_q.transpose(0, 1),
                        beta=0.0, alpha=scale_cpu)
        q = q.view(q_grad.size(0), q_grad.size(1), q_grad.size(2))
        q_grad = q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2))
        weights_q_grad = scale_npu.type_as(input)*torch.mm(input, q_grad)
        return q, weights_q_grad, None, None


class KeyValueLinears(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_k, weights_v):
        ctx.save_for_backward(input, weights_k, weights_v)
        k = torch.addmm(input.view(input.size(0) * input.size(1), input.size(2)),
                        input.view(input.size(0) * input.size(1), input.size(2)), weights_k, beta=0.0, alpha=1.0)
        k = k.view(input.size(0), input.size(1), input.size(2))
        v = torch.addmm(input.view(input.size(0) * input.size(1), input.size(2)),
                        input.view(input.size(0) * input.size(1), input.size(2)), weights_v, beta=0.0, alpha=1.0)
        v = v.view(input.size(0), input.size(1), input.size(2))
        return k.detach(), v.detach()

    @staticmethod
    def backward(ctx, k_grad, v_grad):
        input, weights_k, weights_v = ctx.saved_tensors
        input = input.view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)
        k = torch.addmm(k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)),
                        k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)), weights_k.transpose(0, 1),
                        beta=0.0)
        k_grad = k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)), weights_v.transpose(0, 1), beta=1.0)
        v = v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_k_grad, weights_v_grad


class SelfAttentionLinears(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights_q, weights_k, weights_v, scale_cpu, scale_npu):
        ctx.save_for_backward(input, weights_q, weights_k, weights_v, scale_cpu, scale_npu)
        q = torch.addmm(input.contiguous().view(input.size(0) * input.size(1), input.size(2)),
                        input.contiguous().view(input.size(0) * input.size(1), input.size(2)), weights_q, beta=0.0,
                        alpha=scale_cpu)
        q = q.view(input.size(0), input.size(1), input.size(2))
        k = torch.addmm(input.contiguous().view(input.size(0) * input.size(1), input.size(2)),
                        input.contiguous().view(input.size(0) * input.size(1), input.size(2)), weights_k, beta=0.0,
                        alpha=1.0)
        k = k.view(input.size(0), input.size(1), input.size(2))
        v = torch.addmm(input.contiguous().view(input.size(0) * input.size(1), input.size(2)),
                        input.contiguous().view(input.size(0) * input.size(1), input.size(2)), weights_v, beta=0.0,
                        alpha=1.0)
        v = v.view(input.size(0), input.size(1), input.size(2))
        return q.detach(), k.detach(), v.detach()

    @staticmethod
    def backward(ctx, q_grad, k_grad, v_grad):
        input, weights_q, weights_k, weights_v, scale_cpu, scale_npu = ctx.saved_tensors
        input = input.contiguous().view(input.size(0) * input.size(1), input.size(2)).transpose(0, 1)

        q = torch.addmm(q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)),
                        q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)), weights_q.transpose(0, 1),
                        beta=0.0, alpha=scale_cpu)
        q_grad = q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2))
        weights_q_grad = scale_npu.type_as(input)*torch.mm(input,q_grad)
        k = q.addmm_(k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)), weights_k.transpose(0, 1), beta=1.0)
        k_grad = k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)), weights_v.transpose(0, 1), beta=1.0)
        v = v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_q_grad, weights_k_grad, weights_v_grad, None, None

class StridedBmm1Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = torch.bmm(input1, input2)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_input1 = torch.bmm(grad_output, input2.transpose(1, 2))
        grad_input2 = torch.bmm(grad_output.transpose(1, 2), input1).transpose(1, 2)
        return grad_input1, grad_input2


class StridedBmm2Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = torch.bmm(input1, input2)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_output = grad_output.clone()
        grad_input1 = torch.bmm(grad_output, input2.transpose(1, 2))
        grad_input2 = torch.bmm(input1.transpose(1, 2), grad_output)
        return grad_input1, grad_input2


def query_linear(input: Tensor, weights_q: Tensor, scale_cpu: Tensor, scale_npu: Tensor):
    return QueryLinear.apply(input, weights_q, scale_cpu, scale_npu)

def key_value_linears(input: Tensor, weights_k: Tensor, weights_v: Tensor):
    return KeyValueLinears.apply(input, weights_k, weights_v)

def self_attn_linears(input: Tensor, weights_q: Tensor, weights_k: Tensor, weights_v: Tensor, scale_cpu: Tensor, scale_npu: Tensor):
    return SelfAttentionLinears.apply(input, weights_q, weights_k, weights_v, scale_cpu, scale_npu)

def strided_bmm1(input1: Tensor, input2: Tensor):
    return StridedBmm1Func.apply(input1, input2)

def strided_bmm2(input1: Tensor, input2: Tensor):
    return StridedBmm2Func.apply(input1, input2)

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, seed=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.seed = seed
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.scaling_cpu = Variable(torch.tensor(self.scaling))
        self.scaling_npu = self.scaling_cpu.npu()
        self._mask = torch.empty(0)
        self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias_q = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_k = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_v = Parameter(torch.Tensor(embed_dim))
        else:
            self.register_parameter('in_proj_bias_k', None)
            self.register_parameter('in_proj_bias_q', None)
            self.register_parameter('in_proj_bias_v', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.cache_id = str(id(self))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias_k is not None:
            nn.init.constant_(self.in_proj_bias_q, 0.)
            nn.init.constant_(self.in_proj_bias_k, 0.)
            nn.init.constant_(self.in_proj_bias_v, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask_future_timesteps: bool,
                key_padding_mask: Optional[Tensor],
                incremental_state: Optional[Dict[str, Dict[str, Tensor]]],
                need_weights: bool,
                static_kv: bool):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same, kv_same = self._fast_same_check(query, key, value)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        k = v = query.new_empty(0)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        if qkv_same:
            q, k, v = self_attn_linears(query, self.q_proj_weight,self.k_proj_weight, self.v_proj_weight,
                                        self.scaling_cpu, self.scaling_npu)
        elif kv_same:
            q = query_linear(query, self.q_proj_weight, self.scaling_cpu, self.scaling_npu)
            if not (saved_state is not None and 'prev_key' in saved_state and static_kv):
                k, v = key_value_linears(key, self.k_proj_weight, self.v_proj_weight)
        else:
            q = torch.addmm(query.view(query.size(0) * query.size(1), query.size(2)),
                            query.view(query.size(0) * query.size(1), query.size(2)), self.q_proj_weight, beta=0.0,
                            alpha=self.scaling)
            if not (saved_state is not None and 'prev_key' in saved_state and static_kv):
                k = F.linear(key, self.k_proj_weight, self.in_proj_bias_k)
                v = F.linear(value, self.v_proj_weight, self.in_proj_bias_v)

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).clone().transpose(0, 1).contiguous()
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).clone().transpose(0, 1).contiguous()
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).clone().transpose(0, 1).contiguous()

        attn_weights = strided_bmm1(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                torch.finfo(torch.float32).min,
            ).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training:
            attn_weights, _, _ = torch.npu_dropoutV2(attn_weights, self.seed, p=self.dropout)

        attn = strided_bmm2(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # linear
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = attn_weights.new_empty(0)  # Can't set to None because jit script reasons

        return attn, attn_weights


    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]

        res = F.linear(input, weight, bias)
        return res

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask.size(0) == 0:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new_empty(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Tensor]]]):
        if incremental_state is None or self.cache_id not in incremental_state:
            return {}
        return incremental_state[self.cache_id]

    def _set_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Tensor]]], buffer: Dict[str, Tensor]):
        if incremental_state is not None:
            incremental_state[self.cache_id] = buffer

    def _fast_same_check(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()
        return qkv_same, kv_same

