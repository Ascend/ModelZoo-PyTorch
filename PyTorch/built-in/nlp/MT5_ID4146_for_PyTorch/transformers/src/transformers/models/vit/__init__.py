# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 Huawei Technologies Co., Ltd
#
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
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available, is_vision_available


_import_structure = {
    "configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig", "ViTOnnxConfig"],
}

if is_vision_available():
    _import_structure["feature_extraction_vit"] = ["ViTFeatureExtractor"]

if is_torch_available():
    _import_structure["modeling_vit"] = [
        "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTForImageClassification",
        "ViTForMaskedImageModeling",
        "ViTModel",
        "ViTPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_vit"] = [
        "TFViTForImageClassification",
        "TFViTModel",
        "TFViTPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_vit"] = [
        "FlaxViTForImageClassification",
        "FlaxViTModel",
        "FlaxViTPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig, ViTOnnxConfig

    if is_vision_available():
        from .feature_extraction_vit import ViTFeatureExtractor

    if is_torch_available():
        from .modeling_vit import (
            VIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_vit import TFViTForImageClassification, TFViTModel, TFViTPreTrainedModel

    if is_flax_available():
        from .modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel, FlaxViTPreTrainedModel


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
