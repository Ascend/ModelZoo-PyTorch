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
import json
import sys

f = open(sys.argv[1], 'r')
gpu_nums = int(sys.argv[2])
batch_size = int(sys.argv[3])

epoch = 0
cnt = 0
fps = 0
total_fps = 0
for data in f:
    line = json.loads(data)
    if not 'mode' in line:
        pass
    elif line['mode'] == 'train':
        # ignore first 50 iters
        if line['iter'] > 50:
            cnt = cnt + 1
            fps = fps + gpu_nums * batch_size / line['time']
    elif line['mode'] == 'val':
        epoch = epoch + 1
        print({'epoch': epoch, 'fps': fps / cnt})
        total_fps = total_fps + fps / cnt
        cnt = 0
        fps = 0
print({'fps': total_fps / epoch})
