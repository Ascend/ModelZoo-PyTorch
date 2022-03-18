# -*- coding: utf-8 -*-
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

import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data.dataset import Dataset

from detectron2.data.detection_utils import read_image

ImageTransform = Callable[[torch.Tensor], torch.Tensor]


class ImageListDataset(Dataset):
    """
    Dataset that provides images from a list.
    """

    _EMPTY_IMAGE = torch.empty((0, 3, 1, 1))

    def __init__(
        self,
        image_list: List[str],
        category_list: Union[str, List[str], None] = None,
        transform: Optional[ImageTransform] = None,
    ):
        """
        Args:
            image_list (List[str]): list of paths to image files
            category_list (Union[str, List[str], None]): list of animal categories for
                each image. If it is a string, or None, this applies to all images
        """
        if type(category_list) == list:
            self.category_list = category_list
        else:
            self.category_list = [category_list] * len(image_list)
        assert len(image_list) == len(
            self.category_list
        ), "length of image and category lists must be equal"
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets selected images from the list

        Args:
            idx (int): video index in the video list file
        Returns:
            A dictionary containing two keys:
                images (torch.Tensor): tensor of size [N, 3, H, W] (N = 1, or 0 for _EMPTY_IMAGE)
                categories (List[str]): categories of the frames
        """
        categories = [self.category_list[idx]]
        fpath = self.image_list[idx]
        transform = self.transform

        try:
            image = torch.from_numpy(np.ascontiguousarray(read_image(fpath, format="BGR")))
            image = image.permute(2, 0, 1).unsqueeze(0).float()  # HWC -> NCHW
            if transform is not None:
                image = transform(image)
            return {"images": image, "categories": categories}
        except (OSError, RuntimeError) as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Error opening image file container {fpath}: {e}")

        return {"images": self._EMPTY_IMAGE, "categories": []}

    def __len__(self):
        return len(self.image_list)
