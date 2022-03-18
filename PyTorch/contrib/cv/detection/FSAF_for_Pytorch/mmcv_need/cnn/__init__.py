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

# Copyright (c) Open-MMLab. All rights reserved.
from .alexnet import AlexNet
# yapf: disable
from .bricks import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                     PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS,
                     ContextBlock, Conv2d, Conv3d, ConvAWS2d, ConvModule,
                     ConvTranspose2d, ConvTranspose3d, ConvWS2d,
                     DepthwiseSeparableConvModule, GeneralizedAttention,
                     HSigmoid, HSwish, Linear, MaxPool2d, MaxPool3d,
                     NonLocal1d, NonLocal2d, NonLocal3d, Scale, Swish,
                     build_activation_layer, build_conv_layer,
                     build_norm_layer, build_padding_layer, build_plugin_layer,
                     build_upsample_layer, conv_ws_2d, is_norm)
# yapf: enable
from .resnet import ResNet, make_res_layer
from .utils import (INITIALIZERS, ConstantInit, KaimingInit, NormalInit,
                    PretrainedInit, UniformInit, XavierInit,
                    bias_init_with_prob, caffe2_xavier_init, constant_init,
                    fuse_conv_bn, get_model_complexity_info, initialize,
                    kaiming_init, normal_init, uniform_init, xavier_init)
from .vgg import VGG, make_vgg_layer

__all__ = [
    'AlexNet', 'VGG', 'make_vgg_layer', 'ResNet', 'make_res_layer',
    'constant_init', 'xavier_init', 'normal_init', 'uniform_init',
    'kaiming_init', 'caffe2_xavier_init', 'bias_init_with_prob', 'ConvModule',
    'build_activation_layer', 'build_conv_layer', 'build_norm_layer',
    'build_padding_layer', 'build_upsample_layer', 'build_plugin_layer',
    'is_norm', 'NonLocal1d', 'NonLocal2d', 'NonLocal3d', 'ContextBlock',
    'HSigmoid', 'Swish', 'HSwish', 'GeneralizedAttention', 'ACTIVATION_LAYERS',
    'CONV_LAYERS', 'NORM_LAYERS', 'PADDING_LAYERS', 'UPSAMPLE_LAYERS',
    'PLUGIN_LAYERS', 'Scale', 'get_model_complexity_info', 'conv_ws_2d',
    'ConvAWS2d', 'ConvWS2d', 'fuse_conv_bn', 'DepthwiseSeparableConvModule',
    'Linear', 'Conv2d', 'ConvTranspose2d', 'MaxPool2d', 'ConvTranspose3d',
    'MaxPool3d', 'Conv3d', 'initialize', 'INITIALIZERS', 'ConstantInit',
    'XavierInit', 'NormalInit', 'UniformInit', 'KaimingInit', 'PretrainedInit'
]
