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
import os
import argparse
import time
import tqdm
import cv2
import matplotlib.pylab as plt
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.npu
import numpy as np
import torch.nn as nn
import torch.optim as optim
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from models.posenet import PoseNet
from models.loss_model import MultiTaskLoss
import warnings
import apex
try:
    from apex import amp
except ImportError:
    amp=None

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
    print("========device_id:",NPU_CALCULATE_DEVICE)

#os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"  # choose the available GPUs
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--apex', action='store_true',
                                       help='User apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O1', type=str,
                                       help='For apex mixed precision training'
                                                  'O0 for FP32 training, O1 for mixed precison training.')
args = parser.parse_args()

opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
train_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=16,
                          pin_memory=True)  # num_workers is tuned according to project, too big or small is not good.

soureconfig_val = COCOSourceConfig(opt.hdf5_val_data)
val_data = MyDataset(config, soureconfig_val, shuffle=False, augment=True)  # shuffle in data loader
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=8,
                        pin_memory=True)  # num_workers is tuned according to project, too big or small is not good.

# # ############# for debug  ###################
# if __name__ == '__main__':
#     t0 = time.time()
#     count = 0
#     print(torch.npu.get_device_name(0))
#     torch.backends.cudnn.benchmark = True
#     for epoch in range(20):
#         for bath_id, data_dict in enumerate(train_loader):
#
#             t = data_dict[0].npu(non_blocking=True)  # , non_blocking=True
#             count += opt.batch_size
#             print(bath_id, ' of ', epoch)
#             if count > 500:
#                 break
#     print('**************** ', count / (time.time() - t0))

use_npu = torch.npu.is_available()  # 鍒ゆ柇GPU npu鏄惁鍙敤
best_loss = float('inf')
start_epoch = 0  # 浠0寮濮嬫垨鑰呬粠涓婁竴涓猠poch寮濮

posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=False)
posenet.to(f'npu:{NPU_CALCULATE_DEVICE}')
optimizer = apex.optimizers.NpuFusedSGD(posenet.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)
if args.apex:
    posenet, optimizer = amp.initialize(posenet,optimizer,
                                                          opt_level=args.apex_opt_level,combine_grad=True)

if args.resume:
    print('\nResuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device(f'npu:{NPU_CALCULATE_DEVICE}'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])
    print('\nNetwork weights have been resumed from checkpoint...')

    optimizer.load_state_dict(checkpoint['optimizer_weight'])
    # We must convert the resumed state data of optimizer to gpu
    """It is because the previous training was done on gpu, so when saving the optimizer.state_dict, the stored
     states(tensors) are of npu version. During resuming, when we load the saved optimizer, load_state_dict()
     loads this npu version to cpu. But in this project, we use map_location to map the state tensors to cpu.
     In the training process, we need npu version of state tensors, so we have to convert them to gpu."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.npu()
    print('\nOptimizer has been resumed from checkpoint...')

    best_loss = checkpoint['train_loss']
    print('\n******************** Best loss resumed is :', best_loss, '  ************************')
    start_epoch = checkpoint['epoch'] + 1

# else:
#     print(' # Loading pretrained model # ')
#     posenet.load_state_dict(torch.load(opt.pretrained_model))

criterion = MultiTaskLoss(opt, config)

if use_npu:

    posenet = torch.nn.parallel.DataParallel(posenet.npu())   # , device_ids=[0, 1, 2, 3]
    # module.npu() only move the registered parameters to GPU.
    criterion.npu()

    torch.backends.cudnn.benchmark = True  # 濡傛灉鎴戜滑姣忔璁粌鐨勮緭鍏ユ暟鎹殑size涓嶅彉锛岄偅涔堝紑鍚繖涓氨浼氬姞蹇垜浠殑璁粌閫熷害
    # torch.backends.cudnn.deterministic = True

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, last_epoch=-1)     # 璁剧疆瀛︿範鐜囦笅闄嶇瓥鐣
for i in range(start_epoch):
    #  update the learning rate for start_epoch times
    scheduler.step()


for param in posenet.parameters():
    if param.requires_grad:
        print('Parameters of network: Autograd')
        break


def train(epoch):
    print('\n ############################# Train phase, Epoch: {} #############################'.format(epoch))
    posenet.train()
    train_loss = 0
    scheduler.step()
    print('\nLearning rate at this epoch is: ', optimizer.param_groups[0]['lr'], '\n')

    for batch_idx, target_tuple in enumerate(train_loader):
        if batch_idx == 10:
            pass

        step_time = 0
        start_time = time.time()
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_npu:
            target_tuple = [target_tensor.npu(f'npu:{NPU_CALCULATE_DEVICE}') for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets
        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        optimizer.zero_grad()  # zero the gradient buff

        output_tuple = posenet(images)
        # print(loc_preds.requires_grad)
        # print(conf_preds.requires_grad)
        loss = criterion(output_tuple, target_tuple[1:])
        if loss.item() > 1e6:
            print("Loss is abnormal, drop this batch !")
            continue
        # print(loss.requires_grad)
        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #loss.backward()  # retain_graph=True
        optimizer.step()  # TODO锛氬彲浠ヤ娇鐢ㄧ疮鍔犵殑loss鍙樼浉澧炲ぇbatch size锛屼絾瀵逛簬bn灞傞渶瑕佸噺灏戦粯璁ょ殑momentum
        step_time += (time.time() - start_time)
        fps = opt.batch_size / step_time
        train_loss += loss.item()  # 绱姞鐨刲oss銆銆銆
        # 浣跨敤loss += loss.detach()鏉ヨ幏鍙栦笉闇瑕佹搴﹀洖浼犵殑閮ㄥ垎銆
        # 鎴栬呬娇鐢╨oss.item()鐩存帴鑾峰緱鎵瀵瑰簲鐨刾ython鏁版嵁绫诲瀷锛屼絾鏄粎浠呴檺浜巓nly one element tensors can be converted to Python scalars
        print('########################### Epoch:', epoch, ', --  batch:',  batch_idx, '/', len(train_loader), ',   ',
              'Train loss: %.3f, accumulated average loss: %.3f, step/time(s): %.3f, fps: %.3f ##############################' % (loss.item(), train_loss / (batch_idx + 1), step_time, fps))
        step_time = 0
    global best_loss
    train_loss /= len(train_loader)
    if train_loss < best_loss:
        print('saving...')
        state = {
            # not posenet.state_dict(). then, we don't ge the "module" string to begin with
            'weights': posenet.module.state_dict(),
            'optimizer_weight': optimizer.state_dict(),
            'train_loss': train_loss,
            'epoch': epoch,
        }
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(state, './checkpoints/PoseNet_' + str(epoch) + '_epoch.pth')
        best_loss = train_loss


def test(epoch, show_image=False):
    print('\nTest phase, Epoch: {}'.format(epoch))
    posenet.eval()
    with torch.no_grad():  # will save gpu memory and speed up
        test_loss = 0
        for batch_idx, target_tuple in enumerate(val_loader):
            # images.requires_grad_()
            # loc_targets.requires_grad_()
            # conf_targets.requires_grad_()
            if use_npu:
                target_tuple = [target_tensor.npu(f'npu:{NPU_CALCULATE_DEVICE}') for target_tensor in target_tuple]

            # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
            images, mask_misses, heatmaps = target_tuple # offsets, mask_offset

            # images = Variable(images)
            # loc_targets = Variable(loc_targets)
            # conf_targets = Variable(conf_targets)

            output_tuple = posenet(images)
            loss = criterion(output_tuple, target_tuple[1:])
            test_loss += loss.item()  # 绱姞鐨刲oss
            print('  Test loss : %.3f, accumulated average loss: %.3f' % (loss.item(), test_loss / (batch_idx + 1)))

            # ############### For Debug   ##########
            if show_image:
                image, mask_miss, labels = [v.cpu().numpy() for v in target_tuple] # offsets, mask_offset
                output = output_tuple[-1][0].cpu().numpy()
                # show the generated ground truth
                img = image[0]
                output = output[0].transpose((1, 2, 0))
                img = cv2.resize(img, output.shape[:2], interpolation=cv2.INTER_CUBIC)
                plt.imshow(img[:, :, [2, 1, 0]])  # Opencv image format: BGR
                plt.imshow(output[:, :, 28], alpha=0.5)  # mask_all
                # plt.imshow(mask_offset[:, :, 2], alpha=0.5)  # mask_all
                plt.show()
            #  #######################################


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch, show_image=False)
