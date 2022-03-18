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

import struct
from deep_rl import *


parser = argparse.ArgumentParser()
parser.add_argument('--pth-path', type=str)
parser.add_argument('--state-path', type=str)
parser.add_argument('--outbin-path', type=str)
parser.add_argument('--num', type=int, default=1000)


def to_np(t):
    return t.cpu().detach().numpy()


def bin2np(filepath):
    size = os.path.getsize(filepath)
    res = []
    L = int(size / 4)
    binfile = open(filepath, 'rb')
    for i in range(L):
        data = binfile.read(4)
        num = struct.unpack('f', data)
        res.append(num[0])
    binfile.close()
    dim_res = np.array(res)
    return dim_res


def get_off_action(filepath):
    res = bin2np(filepath)
    res = np.asarray(res)
    action = res.argmax(-1)
    return action


def get_on_action(pthfile, filename):
    config = Config()
    config.task_fn = lambda: Task('BreakoutNoFrameskip-v4')
    config.eval_env = config.task_fn()
    config.history_length = 4
    model = VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    state_dict = torch.load(pthfile, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    network = model
    state = torch.load(filename)
    q = network(state)['q']
    action = to_np(q.argmax(-1))
    return action


if __name__ == "__main__":
    equal = 0
    args = parser.parse_args()
    pth_file = args.pth_path
    output_file = args.state_path
    outbin_file = args.outbin_path
    num = args.num
    for i in range(num):
        off_action = get_off_action('{0}/state{1}_output_0.bin'.format(outbin_file, i))
        on_action = get_on_action(pth_file, '{0}/state{1}.model'.format(output_file, i))
        if off_action == on_action:
            result = 'equal'
            equal+=1
        else:
            result = 'not_equal'
        print("结果：{0}, 在线action:{1}, 离线action:{2}".format(result, on_action, off_action))
        print('精度结果:{}%'.format(equal*100/num))



