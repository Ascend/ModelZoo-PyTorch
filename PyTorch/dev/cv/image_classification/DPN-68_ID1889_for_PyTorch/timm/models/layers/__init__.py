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
from .activations import *
from .adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from .blur_pool import BlurPool2d
from .classifier import ClassifierHead, create_classifier
from .cond_conv2d import CondConv2d, get_condconv_initializer
from .config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from .conv2d_same import Conv2dSame, conv2d_same
from .conv_bn_act import ConvBnAct
from .create_act import create_act_layer, get_act_layer, get_act_fn
from .create_attn import get_attn, create_attn
from .create_conv2d import create_conv2d
from .create_norm_act import get_norm_act_layer, create_norm_act, convert_norm_act
from .drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from .eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from .evo_norm import EvoNormBatch2d, EvoNormSample2d
from .gather_excite import GatherExcite
from .global_context import GlobalContext
from .helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible
from .inplace_abn import InplaceAbn
from .involution import Involution
from .linear import Linear
from .mixed_conv2d import MixedConv2d
from .mlp import Mlp, GluMlp, GatedMlp
from .non_local_attn import NonLocalAttn, BatNonLocalAttn
from .norm import GroupNorm, LayerNorm2d
from .norm_act import BatchNormAct2d, GroupNormAct
from .padding import get_padding, get_same_padding, pad_same
from .patch_embed import PatchEmbed
from .pool2d_same import AvgPool2dSame, create_pool2d
from .squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from .selective_kernel import SelectiveKernel
from .separable_conv import SeparableConv2d, SeparableConvBnAct
from .space_to_depth import SpaceToDepthModule
from .split_attn import SplitAttn
from .split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from .std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from .test_time_pool import TestTimePoolHead, apply_test_time_pool
from .weight_init import trunc_normal_, variance_scaling_, lecun_normal_
