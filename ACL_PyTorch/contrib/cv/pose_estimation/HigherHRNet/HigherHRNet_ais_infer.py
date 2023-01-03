# Copyright 2020 Huawei Technologies Co., Ltd
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
# limitations under the License

import os
import argparse


def ais_infer(bs):

    os.system('mkdir new_bs1_dir')
    os.system('mkdir new_bs1_flip_dir')
    scale_list = [[1024, 512], [512, 1024], [512, 512], [512, 576], [512, 640], [512, 704], [512, 768], [512, 832],
                  [512, 896], [512, 960], [576, 512], [640, 512], [704, 512], [768, 512], [832, 512], [896, 512],
                  [960, 512]]
    for h, w in scale_list:
        os.system('python3 -m ais_bench '
                  '--model=./models/pose_higher_hrnet_w32_512_bs1_dynamic.om '
                  '--input=./prep_output_dir/shape_{1}x{2}/ '
                  '--output=./bs1_dir --outfmt BIN --batchsize {0} --dymHW={1},{2} '
                  '--output_dirname=shape_{1}x{2}'.format(bs, h, w))
        os.system(
            'cp -r bs1_dir/shape_{0}x{1}/*.bin new_bs1_dir/'.format(h, w))

        os.system('python3 -m ais_bench '
                  '--model=./models/pose_higher_hrnet_w32_512_bs1_dynamic.om '
                  '--input=./prep_output_flip_dir/shape_{1}x{2}/ '
                  '--output=./ --outfmt BIN --batchsize {0} --dymHW={1},{2} '
                  '--output_dirname=shape_{1}x{2}'.format(bs, h, w))
        os.system(
            'cp -r bs1_flip_dir/shape_{0}x{1}/*.bin new_bs1_flip_dir/'.format(h, w))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ais_bench')  # task process paramater
    parser.add_argument('--bs', default=1,
                        type=int, help='batchsize')
    args = parser.parse_args()
    ais_infer(args.bs)
