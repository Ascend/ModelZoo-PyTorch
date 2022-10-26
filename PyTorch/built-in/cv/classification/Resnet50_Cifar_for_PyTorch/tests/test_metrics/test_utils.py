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
import pytest
import torch

from mmcls.models.losses.utils import convert_to_one_hot


def ori_convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    assert (torch.max(targets).item() <
            classes), 'Class Index must be less than number of classes'
    one_hot_targets = torch.zeros((targets.shape[0], classes),
                                  dtype=torch.long,
                                  device=targets.device)
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def test_convert_to_one_hot():
    # label should smaller than classes
    targets = torch.tensor([1, 2, 3, 8, 5])
    classes = 5
    with pytest.raises(AssertionError):
        _ = convert_to_one_hot(targets, classes)

    # test with original impl
    classes = 10
    targets = torch.randint(high=classes, size=(10, 1))
    ori_one_hot_targets = torch.zeros((targets.shape[0], classes),
                                      dtype=torch.long,
                                      device=targets.device)
    ori_one_hot_targets.scatter_(1, targets.long(), 1)
    one_hot_targets = convert_to_one_hot(targets, classes)
    assert torch.equal(ori_one_hot_targets, one_hot_targets)


# test cuda version
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_convert_to_one_hot_cuda():
    # test with original impl
    classes = 10
    targets = torch.randint(high=classes, size=(10, 1)).cuda()
    ori_one_hot_targets = torch.zeros((targets.shape[0], classes),
                                      dtype=torch.long,
                                      device=targets.device)
    ori_one_hot_targets.scatter_(1, targets.long(), 1)
    one_hot_targets = convert_to_one_hot(targets, classes)
    assert torch.equal(ori_one_hot_targets, one_hot_targets)
    assert ori_one_hot_targets.device == one_hot_targets.device
