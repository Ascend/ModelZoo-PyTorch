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

import os 
import torch


def _check_directory():
    if not os.path.exists('./models'):
        os.makedirs("./models")


def _get_filename(epoch_id, tag=''):
    if tag == '':
        filename = os.path.join("./models/checkpoint-{}.pth.rar".format(epoch_id))
    else:
        filename = os.path.join("./models/{}-checkpoint-{}.pth.rar".format(tag, epoch_id))
    return filename


def save_checkpoint(model, optimizer, epoch_id, tag=''):
    _check_directory()
    filename = _get_filename(epoch_id, tag)
    torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


def load_model(model, optimizer, epoch_id, device=None, tag=''):
    _check_directory()
    filename = _get_filename(epoch_id, tag)
    if not os.path.exists(filename):
        raise FileNotFoundError("Checkpoint file '{}' not found!".format(filename))
    if device is not None:
        checkpoint = torch.load(filename, map_location=torch.device(device))
    else:
        checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
