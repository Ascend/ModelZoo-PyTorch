# Copyright 2020 Huawei Technologies Co., Ltd
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
# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa
from ..file_utils import DummyObject, requires_backends


class ImageFeatureExtractionMixin(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class BeitFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class CLIPFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class CLIPProcessor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ConvNextFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class DeiTFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class DetrFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ImageGPTFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class LayoutLMv2FeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class LayoutLMv2Processor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class LayoutXLMProcessor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class MaskFormerFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class PerceiverFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class PoolFormerFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class SegformerFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ViltFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ViltProcessor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ViTFeatureExtractor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
