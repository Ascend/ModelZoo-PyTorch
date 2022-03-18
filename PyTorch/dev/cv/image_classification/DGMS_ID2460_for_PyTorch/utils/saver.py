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
import os
import shutil
import torch
from collections import OrderedDict
import glob
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

class Saver(object):

    def __init__(self, args):
        self.args = args
        if args.normal:
            checkname = args.checkname + '_uncompressed'
        else:
            checkname = args.checkname
        self.directory = os.path.join('run', args.dataset, checkname)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            epoch = state['epoch']
            best_top1 = state['best_top1']
            best_top5 = state['best_top5']
            params = state['params']
            bitwidth = state['bits']
            compression_rate = state['CR']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write('Epoch: ' + str(epoch) + "\n")
                f.write('Top1: ' + str(best_top1) + "\n")
                f.write('Top5: ' + str(best_top5) + "\n")
                f.write('#Params: ' + str(params) + "M" + "\n")
                f.write('Bits: ' + str(bitwidth) + "\n")
                f.write('CR: ' + str(compression_rate) + "\n")
            if self.runs:
                previous_acc = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        cnt = 1
                        with open(path, 'r') as f:
                            if cnt == 2: 
                                acc = float(f.readline().split(' ')[-1])
                                previous_acc.append(acc)
                            cnt += 1
                    else:
                        continue
                max_acc = max(previous_acc)
                if best_top1 > max_acc:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'training_configs.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['network'] = self.args.network
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['epoch'] = self.args.epochs
        p['tau'] = self.args.tau
        p['K'] = self.args.K

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
