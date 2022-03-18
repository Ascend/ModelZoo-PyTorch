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

from examples import *

if __name__ == '__main__':
    cf = Config()
    cf.add_argument('--game', type=str, default='BreakoutNoFrameskip-v4')
    cf.add_argument('--use_device', type=str, default='use_npu')
    cf.add_argument('--device_id', type=int, default=0)
    cf.add_argument('--max_steps', type=int, default=2e7)
    cf.add_argument('--save_interval', type=int, default=0)
    cf.add_argument('--eval_interval', type=int, default=0)
    cf.add_argument('--log_interval', type=int, default=1e3)
    cf.add_argument('--tag', type=str, default=None)
    cf.add_argument('--pth_path', type=str, default='null')
    cf.add_argument('--status_path', type=str, default='null')
    cf.merge()

    param = dict(game=cf.game, max_steps=cf.max_steps, save_interval=cf.save_interval, eval_interval=cf.eval_interval,
                 log_interval=cf.log_interval, pth_path=cf.pth_path, status_path=cf.status_path, tag=cf.tag, device_id=cf.device_id,maxremark=dqn_pixel.__name__)

    mkdir('data')
    random_seed()
    select_device(cf.use_device, cf.device_id)
    dqn_pixel(**param)
    exit()