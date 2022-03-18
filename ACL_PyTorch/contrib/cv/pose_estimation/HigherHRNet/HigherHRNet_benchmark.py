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
from gen_dataset_info import scale_list

def benchmark(bs):

    for i in range(len(scale_list)):
        h, w = scale_list[i][0], scale_list[i][1]
        print(h, w)
        os.system('./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size={0} \
                -om_path=./models/pose_higher_hrnet_w32_512_bs{0}_dynamic.om \
                -input_text_path=./prep_bin_{1}x{2}.info -input_width={2} -input_height={1} \
                -output_binary=True -useDvpp=False'.format(bs, h, w))
        os.system(
            'mv result/perf_vision_batchsize_{0}_device_0.txt result/perf_vision_batchsize_{0}_device_0_{1}x{2}.txt'.
                format(bs, h, w))
    os.system('mv result/dumpOutput_device0 result/dumpOutput_device0_bs{}'.format(bs))

    for i in range(len(scale_list)):
        h, w = scale_list[i][0], scale_list[i][1]
        print(h, w)
        os.system('./benchmark.x86_64 -model_type=vision -device_id=0 -batch_size={0} \
                -om_path=./models/pose_higher_hrnet_w32_512_bs{0}_dynamic.om \
                -input_text_path=./prep_bin_flip_{1}x{2}.info -input_width={2} -input_height={1} \
                -output_binary=True -useDvpp=False'.format(bs, h, w))

    os.system('mv result/dumpOutput_device0 result/dumpOutput_device0_bs{}_flip'.format(bs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='benchmark infer')  # task process paramater
    parser.add_argument('--bs', default=1,
                        type=int, help='batchsize')
    args = parser.parse_args()
    benchmark(args.bs)
