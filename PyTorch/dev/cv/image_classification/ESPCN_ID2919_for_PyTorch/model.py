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
# Import Dependencies
import torch
import torch.nn as nn
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class ESPCN(nn.Module):
    def __init__(self, num_channels, scaling_factor):
        """ ESPCN Model class

        :param num_channels (int): Number of channels in input image
        :param scaling_factor (int): Factor to scale-up the input image by
        """

        super(ESPCN, self).__init__()

        # As per paper, 3 conv layers in backbone, adding padding is optional, not mentioned to use in paper
        # SRCNN paper does not recommend using padding, padding here just helps to visualize the scaled up output image
        # Extract input image feature maps
        self.feature_map_layer = nn.Sequential(
            # (f1,n1) = (5, 64)
            nn.Conv2d(in_channels=num_channels, kernel_size=(5, 5), out_channels=64, padding=(2, 2)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh(),
            # (f2,n2) = (3, 32)
            nn.Conv2d(in_channels=64, kernel_size=(3, 3), out_channels=32, padding=(1, 1)),
            # Using "Tanh" activation instead of "ReLU"
            nn.Tanh()
        )

        self.sub_pixel_layer = nn.Sequential(
            # f3 = 3, # output shape: H x W x (C x r**2)
            nn.Conv2d(in_channels=32, kernel_size=(3, 3), out_channels=num_channels * (scaling_factor ** 2), padding=(1, 1)),
            # Sub-Pixel Convolution Layer - PixelShuffle
            # rearranges: H x W x (C x r**2) => rH x rW x C
            nn.PixelShuffle(upscale_factor=scaling_factor)
        )

    def forward(self, x):
        """

        :param x: input image
        :return: model output
        """

        # inputs: H x W x C
        x = self.feature_map_layer(x)
        # output: rH x rW x C
        # r: scale_factor
        out = self.sub_pixel_layer(x)

        return out


if __name__ == '__main__':
    # Print and Test model outputs with a random input Tensor
    sample_input = torch.rand(size=(1, 1, 224, 224))
    print("Input shape: ", sample_input.shape)

    model = ESPCN(num_channels=1, scaling_factor=3)
    print(f"\n{model}\n")

    # Forward pass with sample input
    output = model(sample_input)
    print(f"output shape: {output.shape}")
