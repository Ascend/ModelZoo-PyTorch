# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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

import os
import sys
import time
import torch
from torch import nn
from models.swin import PatchMerging
from utils import format_time


class PatchMergingFixed(PatchMerging):
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor

        # TODO Im2col OP ERROR currently.. Calculate it on CPU temporarily.
        raw_device = x.device
        x = self.patch_merge(x.cpu()).view(b, -1, new_h, new_w).permute(0, 2, 3, 1).to(raw_device)

        x = self.linear(x)

        return x

for k in sys.modules:
    if k == 'models.swin':
        if getattr(sys.modules[k], "PatchMerging", None):
            setattr(sys.modules[k], "PatchMerging", PatchMergingFixed)
            print("PatchMerging has been replaced with PatchMergingFixed for performance optimization.")


try:
	_, term_width_str = os.popen('stty size', 'r').read().split()
except ValueError:
	term_width_str = '80'
term_width = int(term_width_str)


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
total_step_time = 0
def progress_bar(current, total, msg=None):
    global last_time, begin_time, total_step_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
        total_step_time = 0

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    total_step_time += step_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

    return total_step_time/total
