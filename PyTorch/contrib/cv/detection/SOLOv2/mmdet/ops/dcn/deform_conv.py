# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.__version__ >= '1.8':
    import torch_npu
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _single
import math


class ModulatedDeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g, input, weight, offset, bias, stride, padding,
                 dilation, groups, defomable_groups):
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        return g.op(
            'DeformableConv2D',
            input,
            weight,
            offset,
            bias_i=bias,
            strides_i=stride,
            pads_i=padding,
            dilations_i=dilation,
            groups_i=groups,
            defomable_groups_i=defomable_groups)

    @staticmethod
    def forward(ctx,
                input,
                offset_ori,
                mask,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                ):

        input = input.float()
        offset_ori = offset_ori.float()
        mask = mask.float()

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            device = input.device
            bias = torch.zeros(weight.shape[0], device=device)
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]

        offset_x = offset_ori[:, ::2, :, :]
        offset_y = offset_ori[:, 1::2, :, :]
        offset = torch.cat([offset_y, offset_x], dim=1)
        offset_all = torch.cat([offset, mask], dim=1)
        output, offset_out = torch_npu.npu_deformable_conv2d(
            input, weight, offset_all, bias,
            kernel_size=[weight.shape[3], weight.shape[2]],
            stride=[1, 1, ctx.stride, ctx.stride],
            padding=[ctx.padding, ctx.padding, ctx.padding, ctx.padding],
            dilation=[1, 1, ctx.dilation, ctx.dilation],
            groups=ctx.groups, deformable_groups=ctx.deformable_groups,
            modulated=True)
        if weight.requires_grad or mask.requires_grad or offset.requires_grad \
                or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias, offset_out)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias, offset_out = ctx.saved_tensors
        grad_offset = torch.zeros_like(offset)
        offset_all = torch.cat([offset, mask], dim=1)
        grad_input, grad_weight, grad_offset_all, grad_bias = torch_npu.npu_deformable_conv2dbk(
            input, grad_output, offset_out, weight, offset_all,
            kernel_size=[weight.shape[3], weight.shape[2]],
            stride=[1, 1, ctx.stride, ctx.stride],
            padding=[ctx.padding, ctx.padding, ctx.padding, ctx.padding],
            dilation=[1, 1, ctx.dilation, ctx.dilation],
            groups=ctx.groups, deformable_groups=ctx.deformable_groups, modulated=True)
        kernel_hxw = weight.shape[2] * weight.shape[3] * ctx.deformable_groups
        grad_offset[:, 1::2, :, :] = grad_offset_all[:, :kernel_hxw, :, :]
        grad_offset[:, ::2, :, :] = grad_offset_all[:, kernel_hxw:kernel_hxw * 2, :, :]
        grad_mask = grad_offset_all[:, -kernel_hxw:, :, :]
        if not ctx.with_bias:
            grad_bias = None

        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None)

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding -
                      (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding -
                     (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


class ModulatedDeformConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return ModulatedDeformConv2dFunction.apply(x, offset, mask, self.weight, self.bias,
                                                   self.stride, self.padding, self.dilation,
                                                   self.groups, self.deformable_groups)


class ModulatedDeformConvPack(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return ModulatedDeformConv2dFunction.apply(x, offset, mask, self.weight, self.bias,
                                                   self.stride, self.padding, self.dilation,
                                                   self.groups, self.deformable_groups)


DCNv2 = ModulatedDeformConvPack

if __name__ == "__main__":
    x = torch.randn(2, 32, 4, 4)
    model = DCNv2(32, 32, 1)

    torch.npu.set_device(0)
    x = x.npu()
    model = model.npu()

    o = model(x)
    l = o.sum()
    l.backward()
