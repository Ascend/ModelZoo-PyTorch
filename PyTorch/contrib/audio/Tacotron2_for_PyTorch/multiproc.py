# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import subprocess

import torch
import torch_npu


def main():
    argslist = list(sys.argv)[1:]
    world_size = torch.npu.device_count()

    if '--world-size' in argslist:
        argslist[argslist.index('--world-size') + 1] = str(world_size)
    else:
        argslist.append('--world-size')
        argslist.append(str(world_size))

    workers = []

    for i in range(world_size):
        if '--rank' in argslist:
            argslist[argslist.index('--rank') + 1] = str(i)
        else:
            argslist.append('--rank')
            argslist.append(str(i))
        # stdout = None if i == 0 else subprocess.DEVNULL
        worker = subprocess.Popen(
            [str(sys.executable)] + argslist, stdout=None)
        workers.append(worker)

    returncode = 0
    try:
        pending = len(workers)
        while pending > 0:
            for worker in workers:
                try:
                    worker_returncode = worker.wait(1)
                except subprocess.TimeoutExpired:
                    continue
                pending -= 1
                if worker_returncode != 0:
                    if returncode != 1:
                        for worker in workers:
                            worker.terminate()
                    returncode = 1

    except KeyboardInterrupt:
        print('Pressed CTRL-C, TERMINATING')
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()
        raise

    sys.exit(returncode)


if __name__ == "__main__":
    main()
