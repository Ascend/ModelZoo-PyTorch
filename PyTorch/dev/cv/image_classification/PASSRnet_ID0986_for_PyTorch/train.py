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
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.npu
import os
import time
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
import apex
try:
    from apex import amp
except ImportError:
    amp = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='npu:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='data/train/Flickr1024_patches')
    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true', help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
    parser.add_argument('--loss-scale-value', default=1024., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                        help='number of total steps to run')
    return parser.parse_args()


def train(train_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(f'npu:{NPU_CALCULATE_DEVICE}')
    net.apply(weights_init_xavier)
    cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(f'npu:{NPU_CALCULATE_DEVICE}')
    criterion_L1 = L1Loss()
    optimizer = apex.optimizers.NpuFusedAdam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    if cfg.apex:
        net, optimizer = amp.initialize(net, optimizer,
                                               opt_level=cfg.apex_opt_level,
                                               loss_scale=cfg.loss_scale_value,
                                               combine_grad=True)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.n_epochs):
        scheduler.step()
        print(len(train_loader))
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(train_loader):
            if cfg.max_steps and idx_iter >= cfg.max_steps:
                pass
            st_time = time.time()
            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right  = Variable(HR_left).to(f'npu:{NPU_CALCULATE_DEVICE}'), Variable(LR_left).to(f'npu:{NPU_CALCULATE_DEVICE}'), Variable(LR_right).to(f'npu:{NPU_CALCULATE_DEVICE}')

            SR_left, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = net(LR_left, LR_right, is_training=1)

            ### loss_SR
            loss_SR = criterion_mse(SR_left, HR_left)

            ### loss_smoothness
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ### loss_cycle
            Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).to(f'npu:{NPU_CALCULATE_DEVICE}')
            loss_cycle = criterion_L1(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3), Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                         criterion_L1(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3), Identity * V_right_to_left.permute(0, 2, 1, 3))

            ### loss_photometric
            LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b*h,w,w).to(torch.float16), LR_right.permute(0,2,3,1).contiguous().view(b*h, w, c).to(torch.float16))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w).to(torch.float16), LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c).to(torch.float16))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = criterion_L1(LR_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
                          criterion_L1(LR_right * V_right_to_left, LR_left_warped * V_right_to_left)

            ### losses
            loss = loss_SR + 0.005 * (loss_photo + loss_smooth + loss_cycle)

            optimizer.zero_grad()
            if cfg.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            #loss.backward()
            optimizer.step()

            step_time = time.time() - st_time
            print(f"the step{idx_iter} loss:", loss.item(), "step_time", step_time, "fps", cfg.batch_size / step_time)

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path = 'log/x' + str(cfg.scale_factor) + '/', filename='PASSRnet_x' + str(cfg.scale_factor) + '_epoch' + str(idx_epoch + 1) + '.pth.tar')
            psnr_epoch = []
            loss_epoch = []

def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

