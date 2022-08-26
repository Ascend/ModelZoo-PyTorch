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
import glob
import os
import config
import argparse
import json


def task_process(args):
    """[Execute commands for processing]

    Args:
        args ([argparse]): [task process paramater]
    """
    if args.mode == 'change model':
        for i in range(config.center_len):
            h, w = config.center_list[i][0], config.center_list[i][1]
            os.system('rm -rf ctpn_change_{}x{}.onnx'.format(h, w))
        for i in range(config.center_len):
            h, w = config.center_list[i][0], config.center_list[i][1]
            os.system('python3 change_model.py --input_path=ctpn_{}x{}.onnx \
            --output_path=ctpn_change_{}x{}.onnx'.format(h, w, h, w))
    if args.mode == 'preprocess':
        for i in range(config.center_len):
            os.system('rm -rf data/images_bin_{}x{}'.format(config.center_list[i][0], config.center_list[i][1]))
        for i in range(config.center_len):
            os.system('mkdir data/images_bin_{}x{}'.format(config.center_list[i][0], config.center_list[i][1]))
        os.system('python3 ctpn_preprocess.py --src_dir={} --save_path=./data/images_bin'.format(args.src_dir))
    if args.mode == 'ais_infer':
        fps_all = 0
        for i in range(config.center_len):
            h, w = config.center_list[i][0], config.center_list[i][1]

            os.system('python tools/ais-bench_workload/tool/ais_infer/ais_infer.py \
            --model=./ctpn_bs1.om --input=./data/images_bin_{}x{} --dymHW {},{} \
            --batchsize=1 --output=./result/inf_output'.format(h, w, h, w))

            sumary_path = glob.glob('./result/inf_output/*/sumary.json')[0]
            with open(sumary_path, 'r') as f:
                output = json.load(f)
                throughput = output['throughput']
            fps_all = fps_all + throughput * config.center_count[i]
            os.system('rm -f {}'.format(sumary_path))
        os.system('mv ./result/inf_output/*/*.bin ./result/dumpOutput_device0/')
        fps_all = fps_all / config.imgs_len
        if args.machine == 'Ascend310':
            fps_all = fps_all * 4  # 在310上性能要乘以4
        print("===={} performance data====".format(args.machine))
        print('{} bs1 fps:{}'.format(args.machine, fps_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='task process')  # task process paramater
    parser.add_argument('--mode', default='ais_infer',
                        type=str, help='which mode to use')
    parser.add_argument('--src_dir', default='data/Challenge2_Test_Task12_Images',
                        type=str, help='src data dir')
    parser.add_argument('--machine', default='Ascend310P',
                        type=str, help='machine type Ascend310 or Ascend310P')
    args = parser.parse_args()
    task_process(args)
