#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import argparse
import os
from dataset.dataset import get_loader
from solver import Solver
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))

def get_test_info(sal_mode='e'):
    if sal_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = './data/DUTS-TE/Imgs/'
        image_source = './data/DUTS-TE/test.lst'
    elif sal_mode == 'm_r': # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'

    return image_root, image_source

def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--no-npu', dest='npu', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=3)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')

    # Testing settings
    parser.add_argument('--model', type=str, default=None) # Snapshot
    parser.add_argument('--test_fold', type=str, default=None) # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.')
    parser.add_argument('--loss-scale-value', default=128., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
