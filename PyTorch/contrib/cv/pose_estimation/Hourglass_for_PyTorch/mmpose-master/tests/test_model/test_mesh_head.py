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

from mmpose.models import MeshHMRHead
from mmpose.models.mesh_heads.discriminator import SMPLDiscriminator


def test_mesh_hmr_head():
    """Test hmr mesh head."""
    head = MeshHMRHead(in_channels=512)
    head.init_weights()

    input_shape = (1, 512, 8, 8)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    smpl_rotmat, smpl_shape, camera = out
    assert smpl_rotmat.shape == torch.Size([1, 24, 3, 3])
    assert smpl_shape.shape == torch.Size([1, 10])
    assert camera.shape == torch.Size([1, 3])
    """Test hmr mesh head with assigned mean parameters and n_iter """
    head = MeshHMRHead(
        in_channels=512,
        smpl_mean_params='tests/data/smpl/smpl_mean_params.npz',
        n_iter=3)
    head.init_weights()
    input_shape = (1, 512, 8, 8)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    smpl_rotmat, smpl_shape, camera = out
    assert smpl_rotmat.shape == torch.Size([1, 24, 3, 3])
    assert smpl_shape.shape == torch.Size([1, 10])
    assert camera.shape == torch.Size([1, 3])

    # test discriminator with SMPL pose parameters
    # in rotation matrix representation
    disc = SMPLDiscriminator(
        beta_channel=(10, 10, 5, 1),
        per_joint_channel=(9, 32, 32, 16, 1),
        full_pose_channel=(23 * 16, 256, 1))
    pred_theta = (camera, smpl_rotmat, smpl_shape)
    pred_score = disc(pred_theta)
    assert pred_score.shape[1] == 25

    # test discriminator with SMPL pose parameters
    # in axis-angle representation
    pred_theta = (camera, camera.new_zeros([1, 72]), smpl_shape)
    pred_score = disc(pred_theta)
    assert pred_score.shape[1] == 25

    with pytest.raises(TypeError):
        _ = SMPLDiscriminator(
            beta_channel=[10, 10, 5, 1],
            per_joint_channel=(9, 32, 32, 16, 1),
            full_pose_channel=(23 * 16, 256, 1))

    with pytest.raises(ValueError):
        _ = SMPLDiscriminator(
            beta_channel=(10, ),
            per_joint_channel=(9, 32, 32, 16, 1),
            full_pose_channel=(23 * 16, 256, 1))


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run mesh head.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
