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

from mmpose.models import BottomUpHigherResolutionHead, BottomUpSimpleHead


def test_bottom_up_simple_head():
    """test bottom up simple head."""

    with pytest.raises(TypeError):
        # extra
        _ = BottomUpSimpleHead(
            in_channels=512, num_joints=17, with_ae_loss=[True], extra=[])
    # test final_conv_kernel
    with pytest.raises(AssertionError):
        _ = BottomUpSimpleHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True],
            extra={'final_conv_kernel': 0})
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    assert head.final_layer.padding == (1, 1)
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 1})
    head.init_weights()
    assert head.final_layer.padding == (0, 0)
    head = BottomUpSimpleHead(
        in_channels=512, num_joints=17, with_ae_loss=[True])
    head.init_weights()
    assert head.final_layer.padding == (0, 0)
    # test with_ae_loss
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        with_ae_loss=[False],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    # test tag_per_joint
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[False],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 18, 32, 32])
    head = BottomUpSimpleHead(
        in_channels=512,
        num_joints=17,
        num_deconv_layers=0,
        tag_per_joint=False,
        with_ae_loss=[True],
        extra={'final_conv_kernel': 3})
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 18, 32, 32])


def test_bottom_up_higherresolution_head():
    """test bottom up higherresolution head."""

    # test final_conv_kernel
    with pytest.raises(AssertionError):
        _ = BottomUpHigherResolutionHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True, False],
            extra={'final_conv_kernel': 0})
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True])
    head.init_weights()
    assert head.final_layers[0].padding == (1, 1)
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 1},
        cat_output=[True])
    head.init_weights()
    assert head.final_layers[0].padding == (0, 0)
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        cat_output=[True])
    head.init_weights()
    assert head.final_layers[0].padding == (0, 0)
    # test deconv layers
    with pytest.raises(ValueError):
        _ = BottomUpHigherResolutionHead(
            in_channels=512,
            num_joints=17,
            with_ae_loss=[True, False],
            num_deconv_kernels=[1],
            cat_output=[True])
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[4],
        cat_output=[True])
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (0, 0)
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[3],
        cat_output=[True])
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (1, 1)
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        with_ae_loss=[True, False],
        num_deconv_kernels=[2],
        cat_output=[True])
    head.init_weights()
    assert head.deconv_layers[0][0][0].output_padding == (0, 0)
    # test tag_per_joint & ae loss
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=False,
        with_ae_loss=[False, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True])
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 17, 32, 32])
    assert out[1].shape == torch.Size([1, 17, 64, 64])
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=False,
        with_ae_loss=[True, False],
        extra={'final_conv_kernel': 3},
        cat_output=[True])
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 18, 32, 32])
    assert out[1].shape == torch.Size([1, 17, 64, 64])
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[True])
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])
    # cat_output
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[False])
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])
    head = BottomUpHigherResolutionHead(
        in_channels=512,
        num_joints=17,
        tag_per_joint=True,
        with_ae_loss=[True, True],
        extra={'final_conv_kernel': 3},
        cat_output=[False])
    head.init_weights()
    input_shape = (1, 512, 32, 32)
    inputs = _demo_inputs(input_shape)
    out = head([inputs])
    assert out[0].shape == torch.Size([1, 34, 32, 32])
    assert out[1].shape == torch.Size([1, 34, 64, 64])


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
