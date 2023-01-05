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
            os.system('{} change_model.py --input_path={}/ctpn_{}x{}.onnx --output_path={}/ctpn_change_{}x{}.onnx' \
                .format(args.interpreter, args.src_dir, h, w,args.res_dir, h, w)) 
    if args.mode == 'preprocess':
        for i in range(config.center_len):
            os.system('mkdir -p {}_{}x{}'.format(args.res_dir, config.center_list[i][0], config.center_list[i][1]))
        os.system('{} ctpn_preprocess.py --src_dir={} --save_path={}' \
        .format(args.interpreter, args.src_dir, args.res_dir))
    if args.mode == 'ais_infer':
        fps_all = 0
        os.system('mkdir -p {}/inf_output'.format(args.res_dir))
        for i in range(config.center_len):
            h, w = config.center_list[i][0], config.center_list[i][1]

            os.system('{} --model={} --input={}_{}x{} --dymHW {},{} --device {} --batchsize={} --output={}/inf_output' \
            .format(args.interpreter, args.om_path, args.src_dir ,h , w, h, w,args.device, args.batch_size, args.res_dir))

            sumary_path = glob.glob('{}/inf_output/*ary.json'.format(args.res_dir))[0]
            with open(sumary_path, 'r') as f:
                output = json.load(f)
                throughput = output['throughput']  
            fps_all = fps_all + throughput * config.center_count[i]
            os.system('rm -f {}'.format(sumary_path))
        os.system('mv {}/inf_output/*/*.bin {}'.format(args.res_dir, args.res_dir))
        os.system('rm {}/inf_output -rf'.format(args.res_dir))
        fps_all = fps_all / config.imgs_len
        print("====performance data====")
        print('CTPN bs{} models fps:{}'.format(args.batch_size, fps_all))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='task process')  # task process paramater
    parser.add_argument('--interpreter', default='python3',
                        type=str, help='which interpreter to use')
    parser.add_argument('--mode', default='ais_infer',
                        type=str, help='which mode to use')
    parser.add_argument('--src_dir', default='./data/Challenge2_Test_Task12_Images',
                        type=str, help='src data dir')
    parser.add_argument('--res_dir', default='./data/images_bin',
                        type=str, help='res data dir')
    parser.add_argument('--om_path', default='./ctpn_bs1.om',
                        type=str, help='om_path')
    parser.add_argument('--device', default='0',
                        type=str, help='device id')
    parser.add_argument('--batch_size', default='1',
                        type=str, help='batch_size')
    args = parser.parse_args()
    task_process(args)
