# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os

from implOrg.edges_eval_dir import edges_eval_dir
from implOrg.edges_eval_plot import edges_eval_plot


def eval_edge(alg, model_name_list, result_dir, gt_dir, workers=1):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]

    for model_name in model_name_list:
        tic = time.time()
        res_dir = os.path.join(result_dir, model_name)
        print(res_dir)
        edges_eval_dir(res_dir, gt_dir, thin=1, max_dist=0.0075, workers=workers)
        toc = time.time()
        print("TIME: {}s".format(toc - tic))
        edges_eval_plot(res_dir, alg)


if __name__ == '__main__':
    eval_edge("HED", "hed", "NMS_RESULT_FOLDER", "test", 10)
