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
import argparse
import os

import torch
import torch.nn.functional as F

# Lib files
import numpy as np
from torch.utils.data import dataset
import lib.utils as utils
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
from lib.visual3D_temp import non_overlap_padding,test_padding
from lib.losses3D import DiceLoss
from lib.utils.general import prepare_input


from lib.medloaders.brats2018 import MICCAIBraTS2018

from glob import glob



def main():
    args = get_arguments()
    model, optimizer = medzoo.create_model(args)
    batchSz = args.batchSz
    score = 0
    model.eval()
    bin_file_path = args.input_bin
    pth_file_path = args.input_label        

    length = glob(bin_file_path + '/*.bin')
    length1 = glob(pth_file_path + '/*.pth')

    criterion = DiceLoss(classes=args.classes)          

    for s in range(0, len(length)):
        binfile = os.path.join(bin_file_path, str(s)  + '_output_0' + ".bin")
        output = np.fromfile(binfile, dtype=np.float32)
        output = np.reshape(output, (batchSz, 4, 64, 64, 64))
        output = torch.from_numpy(output)

        pthfile = os.path.join(pth_file_path, str(s) + ".pth")
        target = torch.load(pthfile)
        target = torch.from_numpy(target)

        loss_dice, per_ch_score = criterion(output, target)
        avg = np.mean(per_ch_score)
        score += avg
    print("--------score.avg------------", score / len(length))
    return score / len(length)


      
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="brats2018")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--threshold', default=0.00000000001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=5e-3, type=float,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    
    parser.add_argument('--loadData', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='./runs/')
    parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')

    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)


    parser.add_argument('--amp', action='store_true', default=False)

    parser.add_argument('--workers', type=int, default=8)


    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--pretrained',
                default="none",
                type=str, metavar='PATH',
                help='path to pretrained model')
    parser.add_argument('--input_bin', default='none', type=str)
    parser.add_argument('--input_label', default='none', type=str)


    args = parser.parse_args()

    args.save = '../inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args



if __name__ == '__main__':
    main()