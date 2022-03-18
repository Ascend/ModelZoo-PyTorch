# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

import logging

class LRScheduler(object):

    def __init__(self, step_counter=0, base_lr=0.01):
        self.step_counter = step_counter
        self.base_lr = base_lr

    def update(self):
        raise NotImplementedError("must override this")

    def get_lr(self):
        return self.lr

class MultiFactorScheduler(LRScheduler):

    def __init__(self, steps, base_lr=0.01, factor=0.1, step_counter=0):
        super(MultiFactorScheduler, self).__init__(step_counter, base_lr)
        assert isinstance(steps, list) and len(steps) > 0
        for i, _step in enumerate(steps):
            if i != 0 and steps[i] <= steps[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")

        logging.info("Iter %d: start with learning rate: %0.5e (next lr step: %d)" \
                                % (self.step_counter, self.base_lr, steps[0]))
        self.steps = steps
        self.factor = factor
        self.lr = self.base_lr
        self.lr_normalized = self.base_lr
        self.cursor = 0

    def update(self):
        self.step_counter += 1

        if self.cursor >= len(self.steps):
            return self.lr
        while self.steps[self.cursor] < self.step_counter:
            self.lr *= self.factor
            self.cursor += 1
            # message
            if self.cursor >= len(self.steps):
                logging.info("Iter: %d, change learning rate to %0.5e for step [%d:Inf)" \
                                % (self.step_counter-1, self.lr, self.step_counter-1))
                break # return self.lr
            else:
                logging.info("Iter: %d, change learning rate to %0.5e for step [%d:%d)" \
                                % (self.step_counter-1, self.lr, self.step_counter-1, \
                                   self.steps[self.cursor]))
        if self.step_counter < 1500:
            self.lr_normalized = self.lr * (1. + 9.*self.step_counter/1500.) / 10.
        else:
            self.lr_normalized = self.lr
        return self.lr_normalized

    def get_lr(self):
        return self.lr_normalized


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    # test LRScheduler()
    logging.info("testing basic class: LRScheduler()")
    LRScheduler()

    # test MultiFactorScheduler()
    logging.info("testing basic class: MultiFactorScheduler()")
    start_point = 2
    lr_scheduler = MultiFactorScheduler(step_counter=start_point,
                                        base_lr=0.1,
                                        steps=[2, 14, 18],
                                        factor=0.1)
    for i in range(start_point, 22):
        logging.info("id = {}, lr = {:f}".format(i, lr_scheduler.update()))
