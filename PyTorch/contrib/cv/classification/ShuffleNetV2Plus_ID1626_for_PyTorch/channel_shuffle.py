# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    r"""Applies an NPU compatible channel shuffle operation.

        The origin implement is https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py#L21

        In order to avoid contiguous operation which is not efficient on npu, we replaced the original operation
        with a rewrite of the same semantics. Two discontinuous operations are replaced, transpose and chunk.

        .. note::
            Only group=2 is implemented, modify other group scenarios yourself.

        Args:
            in_channels (int): The total number of channels in the input tensors
            groups (int): The number of shuffle groups. Default: 2
            split_shuffle (bool): Whether to execute the chunk after shuffle. Default: True

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`, `(N, C_{in}, L_{in})`
            - Output: :math:`(N, C_{out}, L_{out})`
            
        Examples::
            >>> x1 = torch.randn(2,32,7,7)
            >>> x2 = torch.randn(2,32,7,7)
            >>> m = ChannelShuffle(64, split_shuffle=True)
            >>> output = m(x1, x2)

        """

    def __init__(self, in_channels, groups=2):
        super(ChannelShuffle, self).__init__()
        self.group_len = in_channels // groups

        # init out_channels
        self.out_channels = np.array(list(range(in_channels))).reshape(groups, self.group_len).transpose(1, 0).flatten()
        self.out_channels = torch.from_numpy(self.out_channels).long()

        # init index used in fp & bp
        # Only group=2 is implemented, modify other group scenarios yourself.
        self.fp_index1 = self.out_channels[:self.group_len]
        self.fp_index2 = self.out_channels[self.group_len:]
        self.bp_index = torch.tensor(list(range(0, in_channels, 2)) + list(range(1, in_channels, 2)))

        self.checked = False

    def check_self(self, x):
        r"""Check device equipment between tensors.
        """
        if self.bp_index.device == x.device:
            self.checked = True
            return

        device = x.device
        if str(device).startswith('npu'):
            self.fp_index1 = self.fp_index1.int()
            self.fp_index2 = self.fp_index2.int()
            self.bp_index = self.bp_index.int()

        self.fp_index1 = self.fp_index1.to(device)
        self.fp_index2 = self.fp_index2.to(device)
        self.bp_index = self.bp_index.to(device)


    def forward(self, x):
        if not self.checked:
            self.check_self(x)
        if self.training:
            return IndexSelectHalfImplementation.apply(x, self.fp_index1, self.fp_index2, self.bp_index)
        else:
            return IndexSelectHalfImplementationForward(x, self.fp_index1, self.fp_index2, self.bp_index)

def IndexSelectHalfImplementationForward(x, fp_index1, fp_index2, bp_index):
        return x.index_select(1, fp_index1), x.index_select(1, fp_index2)

class IndexSelectHalfImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fp_index1, fp_index2, bp_index):
        ctx.bp_index = bp_index
        return x.index_select(1, fp_index1), x.index_select(1, fp_index2)

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        grad_output = torch.cat([grad_output1, grad_output2], 1)
        out = grad_output.index_select(1, ctx.bp_index)
        return out, None, None, None, None

def channel_shuffle_torchvision(x, groups=2):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x.chunk(2, 1)

def channel_shuffle_megvii(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

if __name__ == '__main__':
    device = 'cpu'

    if device.startswith('npu'):
        torch.npu.set_device(device)

    channels = 8
    BZ = 2
    H = 1
    W = 1

    # x = torch.randn(BZ, channels, H, W)
    x = torch.arange(BZ*channels*H*W).reshape(BZ, channels, H, W)
    print(x)
    cs_model = ChannelShuffle(channels)

    x = x.to(device)
    cs_model = cs_model.to(device)

    output1 = channel_shuffle_megvii(x)
    print(output1[0])
    output2 = channel_shuffle_torchvision(x)
    print(output2[0])
    output3 = cs_model(x)
    print('output1-output2',sum((i-j).abs().sum() for i, j in zip(output1, output2)))
    print('output2-output3',sum((i-j).abs().sum() for i, j in zip(output2, output3)))
    print('output1-output3',sum((i-j).abs().sum() for i, j in zip(output1, output3)))

