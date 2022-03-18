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

from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " +
                            self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
