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

import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_average_meters(n:int):
    return [AverageMeter() for i in range(n)]


class BlockTimer(object):
    """Measures time used of code block"""

    def __init__(self, device_id='', description=''):
        if device_id in ['', None]:
            device_id = '(unspecified device)'
        if description == '':
            self.start_str = "{} starts.".format(device_id)
            self.finish_formatter = "{} finished. Time used = {{:.3f}}s".format(device_id)
        else:
            self.start_str = "{} starts {}.".format(device_id, description)
            self.finish_formatter = "{} finished {}. Time used = {{:.3f}}s".format(device_id, description)
    
    def __enter__(self):
        print(self.start_str)
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.finish_formatter.format(time.time() - self.start_time))
