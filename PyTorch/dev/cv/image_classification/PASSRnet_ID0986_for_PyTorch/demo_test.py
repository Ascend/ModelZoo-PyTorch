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
from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data/test')
    parser.add_argument('--dataset', type=str, default='KITTI2012')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='npu:0')
    return parser.parse_args()

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(f'npu:{NPU_CALCULATE_DEVICE}')
    cudnn.benchmark = True
    pretrained_dict = torch.load('./log/x' + str(cfg.scale_factor) + '/PASSRnet_x' + str(cfg.scale_factor) + '.pth')
    net.load_state_dict(pretrained_dict)

    psnr_list = []

    with torch.no_grad():
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(f'npu:{NPU_CALCULATE_DEVICE}'), Variable(LR_left).to(f'npu:{NPU_CALCULATE_DEVICE}'), Variable(LR_right).to(f'npu:{NPU_CALCULATE_DEVICE}')
            scene_name = test_loader.dataset.file_list[idx_iter]

            SR_left = net(LR_left, LR_right, is_training=0)
            SR_left = torch.clamp(SR_left, 0, 1)

            psnr_list.append(cal_psnr(HR_left[:,:,:,64:], SR_left[:,:,:,64:]))

            ## save results
            if not os.path.exists('results/'+cfg.dataset):
                os.mkdir('results/'+cfg.dataset)
            if not os.path.exists('results/'+cfg.dataset+'/'+scene_name):
                os.mkdir('results/'+cfg.dataset+'/'+scene_name)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('results/'+cfg.dataset+'/'+scene_name+'/img_0.png')

        ## print results
        print(cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
