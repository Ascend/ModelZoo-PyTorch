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

import pytest
import torch


def test_mse_loss():
    from mmpose.models import build_loss

    # test MSE loss without target weight
    loss_cfg = dict(type='JointsMSELoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    fake_pred = torch.zeros((1, 2, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 2, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))

    with pytest.raises(ValueError):
        loss_cfg = dict(type='JointsOHKMMSELoss')
        loss = build_loss(loss_cfg)
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    with pytest.raises(AssertionError):
        loss_cfg = dict(type='JointsOHKMMSELoss', topk=-1)
        loss = build_loss(loss_cfg)
        fake_pred = torch.zeros((1, 3, 64, 64))
        fake_label = torch.zeros((1, 3, 64, 64))
        assert torch.allclose(
            loss(fake_pred, fake_label, None), torch.tensor(0.))

    loss_cfg = dict(type='JointsOHKMMSELoss', topk=2)
    loss = build_loss(loss_cfg)
    fake_pred = torch.ones((1, 3, 64, 64))
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    loss_cfg = dict(type='JointsOHKMMSELoss', topk=2)
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 64, 64))
    fake_pred[0, 0] += 1
    fake_label = torch.zeros((1, 3, 64, 64))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.5))
