# coding:utf-8
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

import glob
import os
from typing import List, Optional, Tuple

import logging
import numpy as np
import torchvision.transforms.functional as TF
import PIL
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class PathDataset(VisionDataset):
    def __init__(
        self,
        root: List[str],
        loader: None = None,
        transform: Optional[str] = None,
        extra_transform: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super().__init__(root=root)

        PIL.Image.MAX_IMAGE_PIXELS = 256000001

        self.files = []
        for folder in self.root:
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
            )
            self.files.extend(
                sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
            )

        self.transform = transform
        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std

        self.loader = loader

        logger.info(f"loaded {len(self.files)} samples from {root}")

        assert (mean is None) == (std is None)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        path = self.files[idx]

        if self.loader is not None:
            return self.loader(path), None

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        img = TF.to_tensor(img)
        if self.mean is not None and self.std is not None:
            img = TF.normalize(img, self.mean, self.std)
        return img, None
