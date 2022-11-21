# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from yolox.data import COCODataset, ValTransform
import os
import argparse
from tqdm import tqdm
import torch
import sys
sys.path.append('./YOLOX')


def main():
    parser = argparse.ArgumentParser(description="YOLOX Preprocess")
    parser.add_argument('--dataroot', dest='dataroot',
                        help='data root dirname', default='./datasets/COCO',
                        type=str)
    parser.add_argument('--output', dest='output',
                        help='output for prepared data', default='./prep_data',
                        type=str)
    parser.add_argument('--batch',
                        help='validation batch size', default=1,
                        type=int)
    opt = parser.parse_args()

    valdataset = COCODataset(
        data_dir=opt.dataroot,
        json_file='instances_val2017.json',
        name="val2017",
        img_size=(640, 640),
        preproc=ValTransform(legacy=False),
    )
    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {"num_workers": 8, "pin_memory": True,
                         "sampler": sampler, "batch_size": opt.batch}

    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
    if os.path.exists(opt.output):
        os.system("rm-rf " + opt.output)
    else:
        os.system("mkdir " + opt.output)
    for idx, data in enumerate(tqdm(val_loader)):
        data = data[0].detach().numpy()
        output_name = "{:0>12d}.bin".format(idx)
        output_path = os.path.join(opt.output, output_name)
        data.tofile(output_path)


if __name__ == "__main__":
    main()
