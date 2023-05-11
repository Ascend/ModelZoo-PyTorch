# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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
#!/bin/bash

alias cp='cp'
cp -f mmcv_need/_functions.py mmcv/mmcv/parallel/
cp -f mmcv_need/builder.py mmcv/mmcv/runner/optimizer/
cp -f mmcv_need/data_parallel.py mmcv/mmcv/parallel/
cp -f mmcv_need/dist_utils.py mmcv/mmcv/runner/
cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/
cp -f mmcv_need/epoch_based_runner.py mmcv/mmcv/runner/
cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/
cd mmcv
export MMCV_WITH_OPS=1 
export MAX_JOBS=8
python3 setup.py build_ext
python3 setup.py develop
cd ..