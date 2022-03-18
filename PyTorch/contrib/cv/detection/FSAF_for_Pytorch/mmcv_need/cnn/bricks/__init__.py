# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

from .activation import build_activation_layer
from .context_block import ContextBlock
from .conv import build_conv_layer
from .conv2d_adaptive_padding import Conv2dAdaptivePadding
from .conv_module import ConvModule
from .conv_ws import ConvAWS2d, ConvWS2d, conv_ws_2d
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule
from .generalized_attention import GeneralizedAttention
from .hsigmoid import HSigmoid
from .hswish import HSwish
from .non_local import NonLocal1d, NonLocal2d, NonLocal3d
from .norm import build_norm_layer, is_norm
from .padding import build_padding_layer
from .plugin import build_plugin_layer
from .registry import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                       PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS)
from .scale import Scale
from .swish import Swish
from .upsample import build_upsample_layer
from .wrappers import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                       Linear, MaxPool2d, MaxPool3d)

__all__ = [
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
    'build_plugin_layer', 'is_norm', 'HSigmoid', 'HSwish', 'NonLocal1d',
    'NonLocal2d', 'NonLocal3d', 'ContextBlock', 'GeneralizedAttention',
    'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS', 'PADDING_LAYERS',
    'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale', 'ConvAWS2d', 'ConvWS2d',
    'conv_ws_2d', 'DepthwiseSeparableConvModule', 'Swish', 'Linear',
    'Conv2dAdaptivePadding', 'Conv2d', 'ConvTranspose2d', 'MaxPool2d',
    'ConvTranspose3d', 'MaxPool3d', 'Conv3d'
]
