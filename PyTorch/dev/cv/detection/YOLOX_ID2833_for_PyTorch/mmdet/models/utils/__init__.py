
# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
from .brick_wrappers import AdaptiveAvgPool2d, adaptive_avg_pool2d
from .builder import build_linear_layer, build_transformer
from .ckpt_convert import pvt_convert
from .conv_upsample import ConvUpsample
from .csp_layer import CSPLayer
from .gaussian_target import gaussian_radius, gen_gaussian_target
from .inverted_residual import InvertedResidual
from .make_divisible import make_divisible
from .misc import interpolate_as, sigmoid_geometric_mean
from .normed_predictor import NormedConv2d, NormedLinear
from .panoptic_gt_processing import preprocess_panoptic_gt
from .point_sample import (get_uncertain_point_coords_with_randomness,
                           get_uncertainty)
from .positional_encoding import (LearnedPositionalEncoding,
                                  SinePositionalEncoding)
from .res_layer import ResLayer, SimplifiedBasicBlock
from .se_layer import DyReLU, SELayer
from .transformer import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DynamicConv, PatchEmbed, Transformer, nchw_to_nlc,
                          nlc_to_nchw)

__all__ = [
    'ResLayer', 'gaussian_radius', 'gen_gaussian_target',
    'DetrTransformerDecoderLayer', 'DetrTransformerDecoder', 'Transformer',
    'build_transformer', 'build_linear_layer', 'SinePositionalEncoding',
    'LearnedPositionalEncoding', 'DynamicConv', 'SimplifiedBasicBlock',
    'NormedLinear', 'NormedConv2d', 'make_divisible', 'InvertedResidual',
    'SELayer', 'interpolate_as', 'ConvUpsample', 'CSPLayer',
    'adaptive_avg_pool2d', 'AdaptiveAvgPool2d', 'PatchEmbed', 'nchw_to_nlc',
    'nlc_to_nchw', 'pvt_convert', 'sigmoid_geometric_mean',
    'preprocess_panoptic_gt', 'DyReLU',
    'get_uncertain_point_coords_with_randomness', 'get_uncertainty'
]
