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
# limitations under the License.
import sys
sys.path.append(r"./RDN-pytorch")
import argparse
import os
import json
import numpy as np
import torch
from utils import calc_psnr, convert_rgb_to_y, denormalize


def Readfile(preds_file, hr_file, width, height):
    preds = np.loadtxt(preds_file).reshape([3, height*2, width*2])
    hr = np.load(hr_file).reshape([3, height*2, width*2])
    return torch.from_numpy(preds), torch.from_numpy(hr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--label-path', type=str, required=True)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--width', type=int, required=True, default=114)
    parser.add_argument('--height', type=int, required=True, default=114)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()

    writer = open(args.result_path, 'w')
    table_dict = {}
    table_dict["title"] = "Overall statistical evaluation"
    table_dict["value"] = []


    preds = os.listdir(args.pred_path)
    L = len(preds)
    res = []
    for predFile in preds:
        hr = os.path.join(args.label_path, predFile.split('_')[0] + '.npy')
        pred = os.path.join(args.pred_path, predFile)
        preds, hr = Readfile(pred, hr, args.width, args.height)

        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr), dim_order='chw')

        preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

        psnr = calc_psnr(hr_y, preds_y)
        res.append(psnr)
        # print('PSNR of ', predFile, ': {:.2f}'.format(psnr))

    res = np.array(res)
    res = res[np.argsort(-res)]
    Avg_PSNR = 0
    if 'value' not in table_dict.keys():
        print("the item value does not exist!")
    else:
        table_dict["value"].extend(
            [{"key": "Number of images", "value": str(L)}])
        if L != 0:
            Avg_PSNR = np.sum(res) / L
        for i in range(L):
            table_dict["value"].append({"key": "Top" + str(i + 1) + " PSNR",
                                        "value": str(np.around(res[i], decimals=2))})
        table_dict["value"].append({"key": "Avg PSNR",
                                    "value": str(np.around(Avg_PSNR, decimals=2))})
        json.dump(table_dict, writer, indent=4, separators=(', ', ': '))
    writer.close()
    print('Avg_PSNR: {:.2f}'.format(Avg_PSNR))