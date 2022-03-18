# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Filesystem utility functions."""
from __future__ import absolute_import
import os
import errno
import torch
import logging

from ..config import cfg

def save_checkpoint(model, epoch, optimizer=None, lr_scheduler=None, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.TRAIN.MODEL_SAVE_DIR)
    directory = os.path.join(directory, '{}_{}_{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE,
                                                             cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}.pth'.format(str(epoch))
    filename = os.path.join(directory, filename)
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        torch.save(model_state_dict, best_filename)
    else:
        save_state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        if not os.path.exists(filename):
            torch.save(save_state, filename)
            logging.info('Epoch {} model saved in: {}'.format(epoch, filename))

        # remove last epoch
        pre_filename = '{}.pth'.format(str(epoch - 1))
        pre_filename = os.path.join(directory, pre_filename)
        if os.path.exists(pre_filename):
            os.remove(pre_filename)

        # try:
        #     print(torch.npu.synchronize(), "file58")
        #     if os.path.exists(pre_filename):
        #         print(torch.npu.synchronize(), "file60")
        #         os.remove(pre_filename)
        #         print(torch.npu.synchronize(), "file62")
        #     print(torch.npu.synchronize(), "file63")
        # except OSError as e:
        #     print(torch.npu.synchronize(), "file64")
        #     logging.info(e)
        #     print(torch.npu.synchronize(), "file66")

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

