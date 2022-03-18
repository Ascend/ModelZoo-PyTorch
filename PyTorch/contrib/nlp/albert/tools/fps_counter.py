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

import time
from threading import Lock


class FpsCounter:
    """
how to use

fps=FpsCounter()
fps.begin()
code
fps.end()
print(fps.fps())

    """
    def __init__(self):
        self.step_sum = 0
        self.time_sum = 0
        self.t1 = 0
        self.on = False

    def begin(self):
        assert self.on == False, "didnot end last time"
        self.on = True
        self.t1 = time.time_ns()

    def end(self):
        t2 = time.time_ns()
        assert self.on == True, "didnot begin"
        self.time_sum += t2 - self.t1
        self.step_sum += 1
        self.on = False

    def reset(self):
        self.step_sum = 0
        self.time_sum = 0
        self.t1 = 0
        self.on = False

    def fps(self, batch=1, n_device=1):
        if self.step_sum == 0: return 0
        time_avg = self.time_sum / 1e9 / self.step_sum
        return batch * n_device / time_avg


class FpsCounter2:
    def __init__(self, node_num=0):
        self.node_num = node_num
        self.lock = Lock()
        self.step_sum = [0 for i in range(node_num)]
        self.time_sum = [0 for i in range(node_num)]
        self.t1 = [0 for i in range(node_num)]
        self.on = [False for i in range(node_num)]

    def begin(self, node_idx=0):
        assert self.on[node_idx] == False, "didnot end last time"
        self.lock.acquire()
        self.on[node_idx] = True
        self.t1[node_idx] = time.time_ns()
        self.lock.release()

    def end(self, node_idx=0):
        t2 = time.time_ns()
        assert self.on[node_idx] == True, "didnot begin"
        self.lock.acquire()
        self.time_sum[node_idx] += t2 - self.t1[node_idx]
        self.step_sum[node_idx] += 1
        self.on[node_idx] = False
        self.lock.release()

    def reset(self, node_idx=0):
        self.lock.acquire()
        self.step_sum[node_idx] = 0
        self.time_sum[node_idx] = 0
        self.t1[node_idx] = 0
        self.on[node_idx] = False
        self.lock.release()

    def fps(self, batch=1, n_device=1, world_size=0):
        fps = 0
        for i in range(world_size):
            if self.step_sum[i] == 0: continue
            time_avg = self.time_sum[i] / 1e9 / self.step_sum[i]
            fps += batch * n_device / time_avg
        return fps
