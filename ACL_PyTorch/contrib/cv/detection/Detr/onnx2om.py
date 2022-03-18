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

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--auto_tune', default=False)
args = parser.parse_args()

input_shape = [['768,1280,24,40;768,768,24,24;768,1024,24,32'], [1024, 768, 32, 24], [1280, 768, 40, 24],
               [768, 1344, 24, 42], [1344, 768, 42, 24], [1344, 512, 42, 16], [512, 1344, 16, 42]]

to_om = 'atc --framework=5 --model=model/detr_bs{}.onnx -output=auto_om/detr_bs{}_{}_{} ' \
        '--input_shape="inputs:{},3,{},{};mask:{},{},{}" --input_format=ND --soc_version=Ascend310'
to_dyom = 'atc --framework=5 --model=model/detr_bs{}.onnx -output=auto_om/detr_gear_bs{}_{} ' \
          '--input_shape="inputs:{},3,-1,-1;mask:{},-1,-1"  --dynamic_dims="{}" --input_format=ND --soc_version=Ascend310'

if args.auto_tune == True:
    to_om = to_om + ' --auto_tune_mode="RL,GA"'
    to_dyom = to_dyom + ' --auto_tune_mode="RL,GA"'

for i in input_shape:
    if len(i) == 4:
        command = to_om.format(args.batch_size, args.batch_size, i[0], i[1], args.batch_size, i[0], i[1],
                               args.batch_size, i[2], i[3])
        print(command)
        os.system(command)
    else:
        command = to_dyom.format(args.batch_size, args.batch_size,
                                 i[0].split(',')[0], args.batch_size, args.batch_size, i[0])
        print(command)
        os.system(command)