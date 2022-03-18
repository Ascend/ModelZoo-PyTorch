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
import os
import sys
import argparse
from tqdm import tqdm
import torch

sys.path.append('./AlphaPose')
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip


parser = argparse.ArgumentParser(description='AlphaPose Preprocess')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='./AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    type=str)
parser.add_argument('--dataroot', dest='dataroot',
                    help='data root dirname', default='./data/coco',
                    type=str)
parser.add_argument('--output', dest='output',
                    help='output for prepared data', default='prep_data',
                    type=str)
parser.add_argument('--output_flip', dest='output_flip',
                    help='output for prepared fliped data', default='prep_data_fliped',
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default = '-1',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    type=int)
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

opt = parser.parse_args()

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")
os.makedirs(opt.output, exist_ok=True)
os.makedirs(opt.output_flip, exist_ok=True)


def preprocess(cfg, batch_size):
    det_dataset = builder.build_dataset(
        cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False, opt=opt)
    det_loader = torch.utils.data.DataLoader(
        det_dataset, batch_size=batch_size, shuffle=False, num_workers=20, drop_last=False)

    for idx, det_data in tqdm(enumerate(det_loader), dynamic_ncols=True):
        inps = det_data[0].numpy()
        output_name = "{:0>12d}.bin".format(idx)
        output_path = os.path.join(opt.output, output_name)
        inps.tofile(output_path)

        # fliped data
        inps_flip = flip(det_data[0]).numpy()
        output_name_flip = "{:0>12d}.bin".format(idx)
        output_path_flip = os.path.join(opt.output_flip, output_name_flip)
        inps_flip.tofile(output_path_flip)


if __name__ == '__main__':
    config = update_config(opt.cfg)
    preprocess(config, opt.batch)
