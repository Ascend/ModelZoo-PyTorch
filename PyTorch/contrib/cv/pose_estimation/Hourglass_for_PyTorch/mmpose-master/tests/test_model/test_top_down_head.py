# -*- coding: utf-8 -*-
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

import numpy as np
import pytest
import torch

from mmpose.models import (TopDownMSMUHead, TopDownMultiStageHead,
                           TopDownSimpleHead)


def test_top_down_simple_head():
    """Test simple head."""
    with pytest.raises(TypeError):
        # extra
        _ = TopDownSimpleHead(out_channels=3, in_channels=512, extra=[])

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = TopDownSimpleHead(
            out_channels=3, in_channels=512, num_deconv_layers=-1)

    _ = TopDownSimpleHead(out_channels=3, in_channels=512, num_deconv_layers=0)

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = TopDownSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = TopDownSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopDownSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopDownSimpleHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1))

    # test final_conv_kernel
    head = TopDownSimpleHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 3})
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = TopDownSimpleHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})
    assert head.final_layer.padding == (0, 0)
    _ = TopDownSimpleHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 0})

    head = TopDownSimpleHead(out_channels=3, in_channels=512)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 256, 256])

    head = TopDownSimpleHead(
        out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out.shape == torch.Size([1, 3, 32, 32])

    head = TopDownSimpleHead(
        out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out.shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_top_down_multistage_head():
    """Test multistage head."""
    with pytest.raises(TypeError):
        # the number of layers should match
        _ = TopDownMultiStageHead(
            out_channels=3, in_channels=512, num_stages=1, extra=[])

    # test num deconv layers
    with pytest.raises(ValueError):
        _ = TopDownMultiStageHead(
            out_channels=3, in_channels=512, num_deconv_layers=-1)

    _ = TopDownMultiStageHead(
        out_channels=3, in_channels=512, num_deconv_layers=0)

    with pytest.raises(ValueError):
        # the number of layers should match
        _ = TopDownMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the number of kernels should match
        _ = TopDownMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopDownMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_stages=1,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(3, 2, 0))

    with pytest.raises(ValueError):
        # the deconv kernels should be 4, 3, 2
        _ = TopDownMultiStageHead(
            out_channels=3,
            in_channels=512,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, -1))

    with pytest.raises(AssertionError):
        # inputs should be list
        head = TopDownMultiStageHead(out_channels=3, in_channels=512)
        input_shape = (1, 512, 32, 32)
        inputs = _demo_inputs(input_shape)
        out = head(inputs)

    # test final_conv_kernel
    head = TopDownMultiStageHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 3})
    head.init_weights()
    assert head.multi_final_layers[0].padding == (1, 1)
    head = TopDownMultiStageHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 1})
    assert head.multi_final_layers[0].padding == (0, 0)
    _ = TopDownMultiStageHead(
        out_channels=3, in_channels=512, extra={'final_conv_kernel': 0})

    head = TopDownMultiStageHead(out_channels=3, in_channels=512)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert len(out) == 1
    assert out[0].shape == torch.Size([1, 3, 256, 256])

    head = TopDownMultiStageHead(
        out_channels=3, in_channels=512, num_deconv_layers=0)
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 3, 32, 32])

    head.init_weights()


def test_top_down_msmu_head():
    """Test multi-stage multi-unit head."""
    with pytest.raises(AssertionError):
        # inputs should be list
        head = TopDownMSMUHead(
            out_shape=(64, 48), unit_channels=256, num_stages=2, num_units=2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # inputs should be list[list, ...]
        head = TopDownMSMUHead(
            out_shape=(64, 48), unit_channels=256, num_stages=2, num_units=2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [inputs] * 2
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # len(inputs) should equal to num_stages
        head = TopDownMSMUHead(
            out_shape=(64, 48), unit_channels=256, num_stages=2, num_units=2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 2] * 3
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # len(inputs[0]) should equal to num_units
        head = TopDownMSMUHead(
            out_shape=(64, 48), unit_channels=256, num_stages=2, num_units=2)
        input_shape = (1, 256, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 3] * 2
        _ = head(inputs)

    with pytest.raises(AssertionError):
        # input channels should equal to param unit_channels
        head = TopDownMSMUHead(
            out_shape=(64, 48), unit_channels=256, num_stages=2, num_units=2)
        input_shape = (1, 128, 32, 32)
        inputs = _demo_inputs(input_shape)
        inputs = [[inputs] * 2] * 2
        _ = head(inputs)

    head = TopDownMSMUHead(
        out_shape=(64, 48),
        unit_channels=256,
        out_channels=17,
        num_stages=2,
        num_units=2)
    input_shape = (1, 256, 32, 32)
    inputs = _demo_inputs(input_shape)
    inputs = [[inputs] * 2] * 2
    out = head(inputs)
    assert len(out) == 2 * 2
    assert out[0].shape == torch.Size([1, 17, 64, 48])

    head.init_weights()


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
