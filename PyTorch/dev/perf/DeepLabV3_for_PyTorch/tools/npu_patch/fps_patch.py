# Copyright 2023 Huawei Technologies Co., Ltd
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

import copy
import datetime
from typing import Tuple

from mmengine.device import is_cuda_available
from mmengine.runner import LogProcessor


def get_log_after_iter(self, runner, batch_idx: int,
                       mode: str) -> Tuple[dict, str]:
    """Format log string after training, validation or testing epoch.

    Args:
        runner (Runner): The runner of training phase.
        batch_idx (int): The index of the current batch in the current
            loop.
        mode (str): Current mode of runner, train, test or val.

    Return:
        Tuple(dict, str): Formatted log dict/string which will be
        recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
    """
    assert mode in ['train', 'test', 'val']
    cur_iter = self._get_iter(runner, batch_idx=batch_idx)
    # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
    parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                          self.custom_cfg)
    # log_tag is used to write log information to terminal
    # If `self.log_with_hierarchy` is False, the tag is the same as
    # log_tag. Otherwise, each key in tag starts with prefix `train`,
    # `test` or `val`
    log_tag = self._collect_scalars(parsed_cfg, runner, mode)

    if not self.log_with_hierarchy:
        tag = copy.deepcopy(log_tag)
    else:
        tag = self._collect_scalars(parsed_cfg, runner, mode, True)

    # Record learning rate.
    lr_str_list = []
    for key, value in tag.items():
        if key.endswith('lr'):
            key = self._remove_prefix(key, f'{mode}/')
            log_tag.pop(key)
            lr_str_list.append(f'{key}: '
                               f'{value:.{self.num_digits}e}')
    lr_str = ' '.join(lr_str_list)
    # Format log header.
    #  train/val: Epoch [5][5/10]  ...
    #  train: Epoch [5/10000] ... (divided by `max_iter`)
    #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
    if self.by_epoch:
        # Align the iteration log
        dataloader_len = self._get_dataloader_size(runner, mode)
        cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))

        if mode in ['train', 'val']:
            # Right Align the epoch log:
            cur_epoch = self._get_epoch(runner, mode)
            max_epochs = runner.max_epochs
            # 3 means the three characters
            cur_epoch_str = f'[{cur_epoch}]'.rjust(
                len(str(max_epochs)) + 3, ' ')
            tag['epoch'] = cur_epoch
            log_str = (f'Epoch({mode}){cur_epoch_str}'
                       f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            log_str = (f'Epoch({mode}) '
                       f'[{cur_iter_str}/{dataloader_len}]  ')
    else:
        if mode == 'train':
            cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
            log_str = (f'Iter({mode}) '
                       f'[{cur_iter_str}/{runner.max_iters}]  ')
        else:
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter_str = str(batch_idx + 1).rjust(
                len(str(dataloader_len)))
            log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
    # Concatenate lr, momentum string with log header.
    log_str += f'{lr_str}  '
    # If IterTimerHook used in runner, eta, time, and data_time should be
    # recorded.
    if (all(item in log_tag for item in ['time', 'data_time'])
            and 'eta' in runner.message_hub.runtime_info):
        eta = runner.message_hub.get_info('eta')
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        fps = runner.train_dataloader.batch_size * runner. \
            world_size / log_tag["time"]
        log_str += f'eta: {eta_str}  '
        log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                    f'data_time: '
                    f'{log_tag["data_time"]:.{self.num_digits}f}  ')
        log_str += f'FPS: {fps:.3f} '

        # Pop recorded keys
        log_tag.pop('time')
        log_tag.pop('data_time')

    # If cuda is available, the max memory occupied should be calculated.
    if is_cuda_available():
        max_memory = self._get_max_memory(runner)
        log_str += f'memory: {max_memory}  '
        tag['memory'] = max_memory
    # Loop left keys to fill `log_str`.
    if mode in ('train', 'val'):
        log_items = []
        for name, val in log_tag.items():
            if mode == 'val' and not name.startswith('val/loss'):
                continue
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)
    return tag, log_str


LogProcessor.get_log_after_iter = get_log_after_iter
