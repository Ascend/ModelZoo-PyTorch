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
import glob
from tqdm import tqdm
import numpy as np
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append('./workspace/U-2-Net')
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset


WORK_DIR = './workspace/U-2-Net'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess for U-2-Net'
    )
    parser.add_argument('--image_dir', type=str, default='./datasets/ECSSD/images',
                        help='input dataset image dir')
    parser.add_argument('--save_dir', type=str, default='./test_data_ECSSD')
    args = parser.parse_args()
    return args


def preprocess(img_name_lists, save_dirs):
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_lists,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    for idx, data_test in tqdm(enumerate(test_salobj_dataloader)):
        inputs_test = data_test['image'].numpy().astype(np.float32)
        inputs_test.tofile(os.path.join(save_dirs, '{}.bin'.format(idx)))


if __name__ == '__main__':
    args = parse_args()
    image_dir = args.image_dir
    img_name_list = glob.glob(image_dir + os.sep + '*')
    img_name_list.sort()
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    preprocess(img_name_list, save_dir)
