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
import torch.npu
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np
from models import networks
"""!!!!!!!!!!!!!!!修改的地方!!!!!!!!!!!!!!!!!!1"""
LOCAL_RANK = int(os.getenv('LOCAL_RANK', 0))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', 0))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
"""!!!!!!!!!!!!!!!npu修改的地方!!!!!!!!!!!!!!!!!!1"""
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29128'


# python pix2pix_pth2onnx.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --save_onnx True
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    """!!!!!!!!!!!!!!!修改的地方!!!!!!!!!!!!!!!!!!1"""
    """!!!!!!!!!!!!!!!npu修改的地方!!!!!!!!!!!!!!!!!!1"""
    # torch.distributed.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    torch.distributed.init_process_group(backend="hccl", rank=RANK, world_size=WORLD_SIZE)
    print(f"[init] == local rank: {LOCAL_RANK}, global rank: {RANK} , world size: {WORLD_SIZE}")

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    b, c, h, w = 1 ,3, 256, 256
    dummy_input = torch.randn(b, c, h, w, requires_grad=True)
    model.save_onnx(dummy_input)
