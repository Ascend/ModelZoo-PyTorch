#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os

import torch

from concern.config import Configurable, State
from concern.signal_monitor import SignalMonitor


class ModelSaver(Configurable):
    dir_path = State()
    save_interval = State(default=1000)
    signal_path = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        # BUG: signal path should not be global
        self.monitor = SignalMonitor(self.signal_path)

    def maybe_save_model(self, model, epoch, step, logger):
        if step % self.save_interval == 0 or self.monitor.get_signal() is not None:
            self.save_model(model, epoch, step)
            logger.report_time('Saving ')
            logger.iter(step)

    def save_model(self, model, epoch=None, step=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch, step)
                self.save_checkpoint(net, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name('model', epoch, step)
            self.save_checkpoint(model, checkpoint_name)

    def save_checkpoint(self, net, name):
        os.makedirs(self.dir_path, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(self.dir_path, name))

    def make_checkpoint_name(self, name, epoch=None, step=None):
        if epoch is None or step is None:
            c_name = name + '_latest'
        else:
            c_name = '{}_epoch_{}_minibatch_{}'.format(name, epoch, step)
        return c_name
