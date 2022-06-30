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
from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .beamable_mm import BeamableMM
from .bidirectional_multihead_attention import BidirectionalMultiheadSelfAttention
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .grad_multiply import GradMultiply
from .highway import Highway
from .learned_positional_embedding import LearnedPositionalEmbedding
from .multihead_attention import MultiheadAttention
from .bidirectional_multihead_attention import BidirectionalMultiheadSelfAttention
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
__all__ = [
    'AdaptiveInput',
    'AdaptiveSoftmax',
    'BeamableMM',
    'DownsampledMultiHeadAttention',
    'GradMultiply',
    'Highway',
    'LearnedPositionalEmbedding',
    'MultiheadAttention',
    'BidirectionalMultiheadSelfAttention',
    'ScalarBias',
    'SinusoidalPositionalEmbedding',
]
