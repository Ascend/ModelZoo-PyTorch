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
import datetime
import dateutil.tz
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
import random
import torch
if torch.__version__ >= "1.8":
    import torch_npu

from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def get_scale_factor(opt):
    opt.scale_factor = 1.0
    num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    opt.scale_factor_init = opt.scale_factor
    if opt.num_training_scales > 0:
        while num_scales > opt.num_training_scales:
            opt.scale_factor_init = opt.scale_factor_init - 0.01
            num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    return opt.scale_factor_init


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name for training', required=True)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=6)

    parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")
    parser.add_argument('--max_steps', default=None, type=int, metavar='N', help='number of total steps to run')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    if opt.fine_tune:
        _gpu = opt.gpu
        _model_dir = opt.model_dir
        _timestamp = opt.timestamp
        _naive_img = opt.naive_img
        _niter = opt.niter

        opt = functions.load_config(opt)

        opt.gpu = _gpu
        opt.model_dir = _model_dir
        opt.start_scale = opt.train_stages - 1
        opt.timestamp = _timestamp
        opt.fine_tune = True
        opt.naive_img = _naive_img
        opt.niter = _niter

    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.npu.is_available():
        torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
    
    if opt.train_mode == "generation" or opt.train_mode == "retarget" or opt.train_mode == "animation":
        if opt.train_mode == "animation":
            opt.min_size = 20
        from ConSinGAN.training_generation import *
    elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            if opt.model_dir == "":
                print("Model for fine tuning not specified.")
                print("Please use --model_dir to define model location.")
                exit()
            else:
                if not os.path.exists(opt.model_dir):
                    print("Model does not exist: {}".format(opt.model_dir))
                    print("Please specify a valid model.")
                    exit()
            if not os.path.exists(opt.naive_img):
                print("Image for harmonization/editing not found: {}".format(opt.naive_img))
                exit()
        from ConSinGAN.training_harmonization_editing import *
    dir2save = functions.generate_dir2save(opt)
    
    if osp.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # save hyperparameters and code files
    with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
    current_path = os.path.dirname(os.path.abspath(__file__))
    for py_file in glob.glob(osp.join(current_path, "*.py")):
        copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
    copytree(osp.join(current_path, "ConSinGAN"), osp.join(dir2save, "ConSinGAN"))

    # train model
    print("Training model ({})".format(dir2save))
    start = time.time()
    train(opt)
    end = time.time()
    elapsed_time = end - start
    print("Time for training: {} seconds".format(elapsed_time))
