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
import argparse
import re

#center_list = [[140,140],[256,256],[172,114],[128,128],[144,144],[160,240],[180,125],[328,264],[170,256]]
data_path = './preprocess_data'

parser = argparse.ArgumentParser(description='SRGAN get_info script')
parser.add_argument('--data_path', default='./preprocess_data', type=str)
parser.add_argument('--om_path', default='./srgan_dynamic.om', type=str)


def task_process(args):
        file_name = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
        fps_all = 0
        img_count = 0
        for file in file_name:
            print(f'******** start info {file}. *********')
            h = int(file.split('_')[-1])
            w = int(file.split('_')[-2])
            
            os.system('./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size=1 -om_path={} \
            -input_text_path={}/img_{}_{}/img.info -input_width={} -input_height={} \
            -output_binary=True -useDvpp=False'.format(args.om_path,args.data_path,w, h, w, h))
            # 计算相应的fps
            result_txt = 'result/perf_vision_batchsize_1_device_0.txt'
            with open(result_txt, 'r') as f:
                content = f.read()
            txt_data_list = [i.strip() for i in re.findall(r':(.*?),', content.replace('\n', ',') + ',')]
            fps = float(txt_data_list[7].replace('samples/s', '')) * 4
            print(fps)
            fps_all = fps_all + fps
            img_count += 1
        print("====310 performance data====")
        print('310 bs{} fps:{}'.format(result_txt.split('_')[3], fps_all / img_count))


if __name__ == "__main__":
    args = parser.parse_args()
    task_process(args)