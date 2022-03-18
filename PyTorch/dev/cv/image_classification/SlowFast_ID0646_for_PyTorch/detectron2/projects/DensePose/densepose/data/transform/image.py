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
# Copyright (c) Facebook, Inc. and its affiliates.

import torch


class ImageResizeTransform:
    """
    Transform that resizes images loaded from a dataset
    (BGR data in NCHW channel order, typically uint8) to a format ready to be
    consumed by DensePose training (BGR float32 data in NCHW channel order)
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): tensor of size [N, 3, H, W] that contains
                BGR data (typically in uint8)
        Returns:
            images (torch.Tensor): tensor of size [N, 3, H1, W1] where
                H1 and W1 are chosen to respect the specified min and max sizes
                and preserve the original aspect ratio, the data channels
                follow BGR order and the data type is `torch.float32`
        """
        # resize with min size
        images = images.float()
        min_size = min(images.shape[-2:])
        max_size = max(images.shape[-2:])
        scale = min(self.min_size / min_size, self.max_size / max_size)
        images = torch.nn.functional.interpolate(
            images, scale_factor=scale, mode="bilinear", align_corners=False
        )
        return images
