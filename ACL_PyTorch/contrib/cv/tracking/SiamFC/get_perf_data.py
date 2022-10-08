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

import os
import numpy as np
import torch
import sys


total = 5000
exemplar_size = (1, 3, 127, 127)
search_size = (1, 9, 255, 255)


class ExperimentPerformance(object):
    def __init__(self):
        super(ExperimentPerformance, self).__init__()

    def run(self, savepath_e, savepath_s):
        for i in range(total):
            exemplar_input = torch.randn(exemplar_size)
            exemplar_input = np.array(exemplar_input).astype(np.float32)
            exemplar_name = "exemplar{}".format(i)
            exemplar_path = os.path.join(savepath_e, exemplar_name + ".bin")
            exemplar_input.tofile(exemplar_path)

            search_input = torch.randn(search_size)
            search_input = np.array(search_input).astype(np.float32)
            search_name = "search{}".format(i)
            search_path = os.path.join(savepath_s, search_name + ".bin")
            search_input.tofile(search_path)


if __name__ == "__main__":
    save_path_e = sys.argv[1]
    save_path_s = sys.argv[2]
    if not os.path.exists(save_path_e):
        os.makedirs(save_path_e)
    if not os.path.exists(save_path_s):
        os.makedirs(save_path_s)
    e = ExperimentPerformance()
    e.run(save_path_e, save_path_s)
    print("Data For Performance Ready.")
