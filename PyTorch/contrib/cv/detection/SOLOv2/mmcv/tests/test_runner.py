# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

from mock import MagicMock


def test_save_checkpoint():
    try:
        import torch
        from torch import nn
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return

    import mmcv.runner

    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(model=model, batch_processor=lambda x: x)

    with tempfile.TemporaryDirectory() as root:
        runner.save_checkpoint(root)

        latest_path = osp.join(root, 'latest.pth')
        epoch1_path = osp.join(root, 'epoch_1.pth')

        assert osp.exists(latest_path)
        assert osp.exists(epoch1_path)
        assert osp.realpath(latest_path) == epoch1_path

        torch.load(latest_path)


def test_wandb_hook():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return

    import mmcv.runner
    wandb_mock = MagicMock()
    hook = mmcv.runner.hooks.WandbLoggerHook()
    hook.wandb = wandb_mock
    loader = DataLoader(torch.ones((5, 5)))

    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(
        model=model,
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                "accuracy": 0.98
            },
            'num_samples': 5
        })
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)
    wandb_mock.init.assert_called()
    wandb_mock.log.assert_called_with({'accuracy/val': 0.98}, step=5)
    wandb_mock.join.assert_called()
