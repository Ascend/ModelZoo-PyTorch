# coding: utf-8
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import base64
import os
from dataclasses import dataclass, fields, is_dataclass, field
from typing import List, ByteString

import cv2
import numpy as np

from biz.get_dataset_colormap import label_to_color_image

try:
    from typing import get_origin, get_args
except ImportError:
    from typing import _GenericAlias, Generic

    def get_origin(tp):
        if isinstance(tp, _GenericAlias):
            return tp.__origin__
        if tp is Generic:
            return Generic
        return None

    def get_args(tp):
        if isinstance(tp, _GenericAlias) and not tp._special:
            return tp.__args__
        return ()


@dataclass
class DataClassBase:
    def __post_init__(self):
        cls = type(self)
        for f in fields(cls):
            value = getattr(self, f.name)
            if is_dataclass(f.type):
                if not isinstance(value, dict):
                    continue
                new_value = f.type.from_dict(value)
            elif get_origin(f.type) == list:
                ftype = get_args(f.type)[0]
                new_value = [ftype.from_dict(i) for i in value]
            else:
                new_value = value
            setattr(self, f.name, new_value)

    @classmethod
    def from_dict(cls, values):
        known_keys = set(f.name for f in fields(cls))
        return cls(**{
            k: v for k, v in values.items()
            if k in known_keys
        })


@dataclass
class DataItem(DataClassBase):
    file_path: str
    file_name: str
    img: ByteString
    gt: np.ndarray
    crop_size: int


@dataclass
class ImageMask(DataClassBase):
    shape: list
    dataStr: str
    className: list = field(default_factory=list)

    mask: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()

        self.mask = self.data_str_decode(self.dataStr, self.shape)

    def data_str_decode(self, data_str, shape):
        data = base64.b64decode(data_str)
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)


@dataclass
class MxpiImageMask(DataClassBase):
    MxpiImageMask: List[ImageMask]


@dataclass
class PredictResult(DataClassBase):
    mask: np.ndarray

    def save_to_png(self, path, fname):
        color_mask_res = label_to_color_image(self.mask)
        color_mask_res = cv2.cvtColor(color_mask_res.astype(np.uint8),
                                      cv2.COLOR_RGBA2BGR)
        result_path = os.path.join(path, f"{fname}.png")
        cv2.imwrite(result_path, color_mask_res)
