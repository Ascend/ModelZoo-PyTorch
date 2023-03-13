# Copyright 2023 Huawei Technologies Co., Ltd
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
import subprocess


def ais_process(s_list):
    data_sum = 0
    for s in s_list:
        command = 'python3 -m ais_bench --model ./mmpose/hrnet.om --dymDims input:1,3,{},{} --output ./ \
                   --outfmt BIN --loop 1000'.format(s[0], s[1])
        ret = subprocess.getoutput(command)
        data_time = process(ret)
        data_sum += data_time
        print('shape:{}*{} of FPS:'.format(s[0], s[1]))
    print('average of FPS:', data_sum / 36)


def process(file):
    data_line = []
    temp = ''
    result = None
    for f in file:
        if f == '\n':
            data_line.append(temp)
            temp = ''
            continue
        temp += f

    for line in data_line:
        if 'throughput' in line:
            result = line.split(':')[1]
            break
    return result


if __name__ == '__main__':
    shape_list = [[512, 768], [512, 832], [512, 704], [512, 896], [512, 640], [512, 960], [512, 512], [512, 1152],
                  [512, 576], [512, 1024], [512, 1088], [512, 1280], [512, 1984], [512, 1792], [512, 1536],
                  [512, 2048], [512, 2112], [512, 1216], [512, 1856], [512, 1344], [512, 1728], [512, 1920],
                  [512, 1472], [512, 1600], [512, 1408], [768, 512], [704, 512], [576, 512], [832, 512], [640, 512],
                  [960, 512], [960, 512], [896, 512], [1152, 512], [1088, 512], [1024, 512], [1344, 512]]
    ais_process(shape_list)
