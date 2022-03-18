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
from typing import BinaryIO, Dict, Union
import torch


def normalized_coords_transform(x0, y0, w, h):
    """
    Coordinates transform that maps top left corner to (-1, -1) and bottom
    right corner to (1, 1). Used for torch.grid_sample to initialize the
    grid
    """

    def f(p):
        return (2 * (p[0] - x0) / w - 1, 2 * (p[1] - y0) / h - 1)

    return f


class DensePoseTransformData(object):

    # Horizontal symmetry label transforms used for horizontal flip
    MASK_LABEL_SYMMETRIES = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
    # fmt: off
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]  # noqa
    # fmt: on

    def __init__(self, uv_symmetries: Dict[str, torch.Tensor], device: torch.device):
        self.mask_label_symmetries = DensePoseTransformData.MASK_LABEL_SYMMETRIES
        self.point_label_symmetries = DensePoseTransformData.POINT_LABEL_SYMMETRIES
        self.uv_symmetries = uv_symmetries
        self.device = torch.device("cpu")

    def to(self, device: torch.device, copy: bool = False) -> "DensePoseTransformData":
        """
        Convert transform data to the specified device

        Args:
            device (torch.device): device to convert the data to
            copy (bool): flag that specifies whether to copy or to reference the data
                in case the device is the same
        Return:
            An instance of `DensePoseTransformData` with data stored on the specified device
        """
        if self.device == device and not copy:
            return self
        uv_symmetry_map = {}
        for key in self.uv_symmetries:
            uv_symmetry_map[key] = self.uv_symmetries[key].to(device=device, copy=copy)
        return DensePoseTransformData(uv_symmetry_map, device)

    @staticmethod
    def load(io: Union[str, BinaryIO]):
        """
        Args:
            io: (str or binary file-like object): input file to load data from
        Returns:
            An instance of `DensePoseTransformData` with transforms loaded from the file
        """
        import scipy.io

        uv_symmetry_map = scipy.io.loadmat(io)
        uv_symmetry_map_torch = {}
        for key in ["U_transforms", "V_transforms"]:
            uv_symmetry_map_torch[key] = []
            map_src = uv_symmetry_map[key]
            map_dst = uv_symmetry_map_torch[key]
            for i in range(map_src.shape[1]):
                map_dst.append(torch.from_numpy(map_src[0, i]).to(dtype=torch.float))
            uv_symmetry_map_torch[key] = torch.stack(map_dst, dim=0)
        transform_data = DensePoseTransformData(uv_symmetry_map_torch, device=torch.device("cpu"))
        return transform_data
