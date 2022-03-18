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
parser.add_argument('--img_path', default='img_file')
parser.add_argument('--mask_path', default='mask_file')
parser.add_argument('--out_put', default='out_put')
parser.add_argument('--result', default='result')
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()

if not os.path.exists(args.out_put):
    os.mkdir(args.out_put)
if not os.path.exists(args.result):
    os.mkdir(args.result)

shape_3 = [[768, 1280, 24, 40], [768, 768, 24, 24], [768, 1024, 24, 32]]
shape_6 = [[1024, 768, 32, 24], [1280, 768, 40, 24], [768, 1344, 24, 42], [1344, 768, 42, 24], [1344, 512, 32, 42],
           [512, 1344, 16, 42]]
print(args)
if args.batch_size == 1:
    for i in shape_3:
        command = 'tools/msame/out/msame --model "auto_om/detr_gear_bs1_768.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}" ' \
                  '--dymDims "inputs:1,3,{},{};mask:1,{},{}" --outfmt BIN'. \
            format(args.img_path, i[0], i[1], args.mask_path, i[0], i[1], args.out_put, i[0], i[1], int(i[0] / 32),
                   int(i[1] / 32))
        print(command)
        os.system(command)
    for i in shape_6:
        command = 'tools/msame/out/msame --model "auto_om/detr_bs1_{}_{}.om" --input "{}/{}_{},{}/{}_{}_mask" --output "{}"  --outfmt BIN'.format(
            i[0], i[1], args.img_path, i[0], i[1], args.mask_path, i[0], i[1], args.out_put)
        print(command)
        os.system(command)
    mv_command = 'mv {}/*/* {}'.format(args.out_put, args.result)
    os.system(mv_command)
elif args.batch_size == 4:
    print(4)
    for i in shape_3:
        command = 'tools/msame/out/msame --model "auto_om/detr_gear_bs4_768.om"  --output "{}" --dymDims "inputs:4,3,{},{};mask:4,{},{}" ' \
                  '--outfmt BIN --loop 20'.format(args.out_put, i[0], i[1], int(i[0] / 32), int(i[1] / 32))
        print(command)
        os.system(command)
    for i in shape_6:
        command = 'tools/msame/out/msame --model "auto_om/detr_bs4_{}_{}.om" --output "{}"  --outfmt BIN --loop 20'. \
            format(i[0], i[1], args.out_put)
        print(command)
        os.system(command)
