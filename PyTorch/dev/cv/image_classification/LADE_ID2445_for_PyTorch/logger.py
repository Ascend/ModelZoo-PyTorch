#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import yaml
import csv
import h5py
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.cfg_file = os.path.join(self.logdir, 'cfg.yaml')
        self.acc_file = os.path.join(self.logdir, 'acc.csv')
        self.loss_file = os.path.join(self.logdir, 'loss.csv')
        self.ws_file = os.path.join(self.logdir, 'ws.h5')
        self.acc_keys = None
        self.loss_keys = None
        self.logging_ws = False

    def log_cfg(self, cfg):
        print('===> Saving cfg parameters to: ', self.cfg_file)
        with open(self.cfg_file, 'w') as f:
            yaml.dump(cfg, f)

    def log_acc(self, accs):
        if self.acc_keys is None:
            self.acc_keys = [k for k in accs.keys()]
            with open(self.acc_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writeheader()
                writer.writerow(accs)
        else:
            with open(self.acc_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writerow(accs)

    def log_loss(self, losses):
        # valid_losses = {k: v for k, v in losses.items() if v is not None}
        valid_losses = losses
        if self.loss_keys is None:
            self.loss_keys = [k for k in valid_losses.keys()]
            with open(self.loss_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writeheader()
                writer.writerow(valid_losses)
        else:
            with open(self.loss_file, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writerow(valid_losses)
    
    def log_ws(self, e, ws):
        mode = 'a' if self.logging_ws else 'w'
        self.logging_ws = True
        
        key = 'Epoch{:02d}'.format(e)
        with h5py.File(self.ws_file, mode) as f:
            g = f.create_group(key)
            for k, v in ws.items():
                g.create_dataset(k, data=v)
        