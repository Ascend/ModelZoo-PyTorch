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
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import apex.optimizers as apex_optim
import torch.distributed as dist
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data.dataloader import DataLoader
from models.posenet import Network
from models.loss_model import MultiTaskLoss
import warnings

try:
    import apex.optimizers as apex_optim
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--freeze', action='store_true', default=False,
                    help='freeze the pre-trained layers before output layers')
parser.add_argument('--warmup', action='store_true', default=True, help='using warm-up learning rate')
parser.add_argument('--checkpoint_path', '-p', default='link2checkpoints_distributed', help='save path')
parser.add_argument('--max_grad_norm', default=10, type=float,
                    help=("If the norm of the gradient vector exceeds this, "
                          "re-normalize it to have the norm equal to max_grad_norm"))
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--sync_bn', action='store_true', default=True,
                    help='enabling apex sync BN.')  # 鏃犺Е鍙戜负false锛 -s 瑙﹀彂涓簍rue
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)  # '1.0'
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N', help='print frequency (default: 10)')

# ##############################################################################################################
# ###################################  Setup for some configurations ###########################################
# ##############################################################################################################

torch.backends.cudnn.benchmark = True  # 濡傛灉鎴戜滑姣忔璁粌鐨勮緭鍏ユ暟鎹殑size涓嶅彉锛岄偅涔堝紑鍚繖涓氨浼氬姞蹇垜浠殑璁粌閫熷害
use_npu = torch.npu.is_available()

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
train_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader

soureconfig_val = COCOSourceConfig(opt.hdf5_val_data)
val_data = MyDataset(config, soureconfig_val, shuffle=False, augment=False)  # shuffle in data loader

best_loss = float('inf')
start_epoch = 0  # 浠0寮濮嬫垨鑰呬粠涓婁竴涓猠poch寮濮

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
if args.distributed:
    args.gpu = args.local_rank
    torch.npu.set_device('npu')
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    torch.distributed.init_process_group(backend='hccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()  # 鑾峰彇鍒嗗竷寮忚缁冪殑杩涚▼鏁
    print("World Size is :", args.world_size)

assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

model = Network(opt, config, dist=True, bn=True)

if args.sync_bn:  # 鐢ㄧ疮璁oss鏉ヨ揪鍒皊ync bn 鏄笉鏄洿濂斤紝鏇存敼bn鐨刴omentum澶у皬
    #  This should be done before model = DDP(model, delay_allreduce=True),
    #  because DDP needs to see the finalized model parameters
    # We rely on torch distributed for synchronization between processes. Only DDP support the apex sync_bn now.
    import apex

    print("Using apex synced BN.")
    model = apex.parallel.convert_syncbn_model(model)

# It should be called before constructing optimizer if the module will live on GPU while being optimized.
model.npu()

for param in model.parameters():
    if param.requires_grad:
        print('Parameters of network: Autograd')
        break

# ##############################################################################################################
# ######################################## Froze some layers to fine-turn the model  ########################
# ##############################################################################################################
if args.freeze:
    for name, param in model.named_parameters():  # 甯︽湁鍙傛暟鍚嶇殑妯″瀷鐨勫悇涓眰鍖呭惈鐨勫弬鏁伴亶鍘
        if 'out' or 'merge' or 'before_regress' in name:  # 鍒ゆ柇鍙傛暟鍚嶅瓧绗︿覆涓槸鍚﹀寘鍚煇浜涘叧閿瓧
            continue
        param.requires_grad = False
# #############################################################################################################

# Actual working batch size on multi-GPUs is 4 times bigger than that on one GPU
# fixme: add up momentum if the batch grows?
# fixme: change weight_decay?
#    nesterov = True
# optimizer = apex_optim.FusedSGD(filter(lambda p: p.requires_grad, model.parameters()),
#                                 lr=opt.learning_rate * args.world_size, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=opt.learning_rate * args.world_size, momentum=0.9, weight_decay=5e-4)
# optimizer = apex_optim.FusedAdam(model.parameters(), lr=opt.learning_rate * args.world_size, weight_decay=1e-4)


# 璁剧疆瀛︿範鐜囦笅闄嶇瓥鐣, extract the "bare"  Pytorch optimizer before Apex wrapping.
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4, last_epoch=-1)


# Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
# for convenient interoperation with argparse.
# For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
# This must be done AFTER the call to amp.initialize.
model, optimizer = amp.initialize(model, optimizer,
                                  opt_level=args.opt_level,
                                  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                  loss_scale=args.loss_scale,
                                  combine_grad=True)  # Dynamic loss scaling is used by default.

if args.distributed:
    # By default, apex.parallel.DistributedDataParallel overlaps communication with computation in the backward pass.
    # model = DDP(model)
    # delay_allreduce delays all communication to the end of the backward pass.
    # DDP妯″潡鍚屾椂涔熻绠楁暣浣撶殑骞冲潎姊害, 杩欐牱鎴戜滑灏变笉闇瑕佸湪璁粌姝ラ璁＄畻骞冲潎姊害銆
    model = DDP(model, delay_allreduce=True)

# ###################################  Resume from checkpoint ###########################################
if args.resume:
    # Use a local scope to avoid dangling references
    # dangling references: a variable that refers to an object that was deleted prematurely
    def resume():
        if os.path.isfile(opt.ckpt_path):
            print('Resuming from checkpoint ...... ')
            checkpoint = torch.load(opt.ckpt_path,
                                    map_location=torch.device('npu'))  # map to cpu to save the gpu memory

            # #################################################
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            # # #################################################
            for k, v in checkpoint['weights'].items():
                # Exclude the regression layer by commenting the following code when we change the output dims!
                # if 'out' or 'merge' or 'before_regress'in k:
                #     continue
                name = 'module.' + k  # add prefix 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)  # , strict=False
            # # #################################################
            # model.load_state_dict(checkpoint['weights'])  # 鍔犲叆浠栦汉璁粌鐨勬ā鍨嬶紝鍙兘闇瑕佸拷鐣ラ儴鍒嗗眰锛屽垯strict=False
            print('Network weights have been resumed from checkpoint...')

            # amp.load_state_dict(checkpoint['amp'])
            # print('AMP loss_scalers and unskipped steps have been resumed from checkpoint...')

            # ############## We must convert the resumed state data of optimizer to gpu  ##############
            # """It is because the previous training was done on gpu, so when saving the optimizer.state_dict, the stored
            #  states(tensors) are of npu version. During resuming, when we load the saved optimizer, load_state_dict()
            #  loads this npu version to cpu. But in this project, we use map_location to map the state tensors to cpu.
            #  In the training process, we need npu version of state tensors, so we have to convert them to gpu."""
            optimizer.load_state_dict(checkpoint['optimizer_weight'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.npu()
            print('Optimizer has been resumed from checkpoint...')
            global best_loss, start_epoch  # global declaration. otherwise best_loss and start_epoch can not be changed
            best_loss = checkpoint['train_loss']
            print('******************** Best loss resumed is :', best_loss, '  ************************')
            start_epoch = checkpoint['epoch'] + 1
            print("========> Resume and start training from Epoch {} ".format(start_epoch))
            del checkpoint
        else:
            print("========> No checkpoint found at '{}'".format(opt.ckpt_path))


    resume()

train_sampler = None
val_sampler = None
# Restricts data loading to a subset of the dataset exclusive to the current process
# Create DistributedSampler to handle distributing the dataset across nodes when training 鍒涘缓鍒嗗竷寮忛噰鏍峰櫒鏉ユ帶鍒惰缁冧腑鑺傜偣闂寸殑鏁版嵁鍒嗗彂
# This can only be called after distributed.init_process_group is called 杩欎釜鍙兘鍦 distributed.init_process_group 琚皟鐢ㄥ悗璋冪敤
# 杩欎釜瀵硅薄鎺у埗杩涘叆鍒嗗竷寮忕幆澧冪殑鏁版嵁闆嗕互纭繚妯″瀷涓嶆槸瀵瑰悓涓涓瓙鏁版嵁闆嗚缁冿紝浠ヨ揪鍒拌缁冪洰鏍囥
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

# 鍒涘缓鏁版嵁鍔犺浇鍣紝鍦ㄨ缁冨拰楠岃瘉姝ラ涓杺鏁版嵁
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                           num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=2, pin_memory=True, sampler=val_sampler, drop_last=True)

for param in model.parameters():
    if param.requires_grad:
        print('Parameters of network: Autograd')
        break


# #  Update the learning rate for start_epoch times
# for i in range(start_epoch):
#     scheduler.step()


def train(epoch):
    print('\n ############################# Train phase, Epoch: {} #############################'.format(epoch))
    torch.npu.empty_cache()
    model.train()
    # DistributedSampler 涓褰曠洰鍓嶇殑 epoch 鏁帮紝 鍥犱负閲囨牱鍣ㄦ槸鏍规嵁 epoch 鏉ュ喅瀹氬浣曟墦涔卞垎閰嶆暟鎹繘鍚勪釜杩涚▼
    if args.distributed:
        train_sampler.set_epoch(epoch)
    # scheduler.step()  use 'adjust learning rate' instead

    # adjust_learning_rate_cyclic(optimizer, epoch, start_epoch)  # start_epoch
    print('\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0]['lr'])  # scheduler.get_lr()[0]

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, target_tuple in enumerate(train_loader):
        # # ##############  Use schedule step or fun of 'adjust learning rate' #####################
        adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader), use_warmup=args.warmup)
        # print('\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0]['lr'])  # scheduler.get_lr()[0]
        # # ##########################################################
        if use_npu:
            #  杩欏厑璁稿紓姝 GPU 澶嶅埗鏁版嵁涔熷氨鏄璁＄畻鍜屾暟鎹紶杈撳彲浠ュ悓鏃惰繘.
            target_tuple = [target_tensor.npu('npu') for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets
        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)
        optimizer.zero_grad()  # zero the gradient buff
        loss = model(target_tuple)

        if loss.item() > 2e5:  # try to rescue the gradient explosion
            print("\nOh My God ! \nLoss is abnormal, drop this batch !")
            continue

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)  # fixme: 鍙兘鏄繖涓殑闂鍚楋紵
        optimizer.step()  # TODO锛氬彲浠ヤ娇鐢ㄧ疮鍔犵殑loss鍙樼浉澧炲ぇbatch size锛屼絾瀵逛簬bn灞傞渶瑕佸噺灏戦粯璁ょ殑momentum

        # train_loss += loss.item()  # 绱姞鐨刲oss !
        # 浣跨敤loss += loss.detach()鏉ヨ幏鍙栦笉闇瑕佹搴﹀洖浼犵殑閮ㄥ垎銆
        # 鎴栬呬娇鐢╨oss.item()鐩存帴鑾峰緱鎵瀵瑰簲鐨刾ython鏁版嵁绫诲瀷锛屼絾鏄粎浠呴檺浜巓nly one element tensors can be converted to Python scalars
        if batch_idx % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.
            # print 浼氳Е鍙慳llreduce锛岃岃繖涓搷浣滄瘮杈冭垂鏃
            if args.distributed:
                # We manually reduce and average the metrics across processes. In-place reduce tensor.
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
            torch.npu.synchronize()  # 鍥犱负鎵鏈塆PU鎿嶄綔鏄紓姝ョ殑锛屽簲绛夊緟褰撳墠璁惧涓婃墍鏈夋祦涓殑鎵鏈夋牳蹇冨畬鎴愶紝娴嬭瘯鐨勬椂闂存墠姝ｇ‘
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:  # Print them in the Process 0
                print('==================> Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f}) <================ \t'.format(
                    epoch, batch_idx, len(train_loader),
                    args.world_size * opt.batch_size / batch_time.val,
                    args.world_size * opt.batch_size / batch_time.avg,
                    batch_time=batch_time,
                    loss=losses))

    global best_loss
    # DistributedSampler鎺у埗杩涘叆鍒嗗竷寮忕幆澧冪殑鏁版嵁闆嗕互纭繚妯″瀷涓嶆槸瀵瑰悓涓涓瓙鏁版嵁闆嗚缁冿紝浠ヨ揪鍒拌缁冪洰鏍囥
    # train_loss /= (len(train_loader))  # Each GPU process can only see 1/(world_size) training samples per epoch

    if args.local_rank == 0:
        # Write the log file each epoch.
        os.makedirs(checkpoint_path, exist_ok=True)
        logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
        logger.write('\nEpoch {}\ttrain_loss: {}'.format(epoch, losses.avg))  # validation鏃朵笉瑕乗n鎹㈣
        logger.flush()
        logger.close()

        if losses.avg < float('inf'):  # < best_loss
            # Update the best_loss if the average loss drops
            best_loss = losses.avg
            print('\nSaving model checkpoint...\n')
            state = {
                # not posenet.state_dict(). then, we don't ge the "module" string to begin with
                'weights': model.module.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                # 'amp': amp.state_dict(),
                'train_loss': losses.avg,
                'epoch': epoch
            }
            torch.save(state, './' + checkpoint_path + '/PoseNet_' + str(epoch) + '_epoch.pth')


def test(epoch):
    print('\n ############################# Test phase, Epoch: {} #############################'.format(epoch))
    model.eval()
    # DistributedSampler 涓褰曠洰鍓嶇殑 epoch 鏁帮紝 鍥犱负閲囨牱鍣ㄦ槸鏍规嵁 epoch 鏉ュ喅瀹氬浣曟墦涔卞垎閰嶆暟鎹繘鍚勪釜杩涚▼
    # if args.distributed:
    #     val_sampler.set_epoch(epoch)  # 楠岃瘉闆嗗お灏忥紝涓嶅4涓垝鍒
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, target_tuple in enumerate(val_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_npu:
            #  杩欏厑璁稿紓姝 GPU 澶嶅埗鏁版嵁涔熷氨鏄璁＄畻鍜屾暟鎹紶杈撳彲浠ュ悓鏃惰繘.
            target_tuple = [target_tensor.npu('npu') for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets

        with torch.no_grad():
            _, loss = model(target_tuple)

        if args.distributed:
            # We manually reduce and average the metrics across processes. In-place reduce tensor.
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
        torch.npu.synchronize()  # 鍥犱负鎵鏈塆PU鎿嶄綔鏄紓姝ョ殑锛屽簲绛夊緟褰撳墠璁惧涓婃墍鏈夋祦涓殑鎵鏈夋牳蹇冨畬鎴愶紝娴嬭瘯鐨勬椂闂存墠姝ｇ‘
        batch_time.update((time.time() - end))
        end = time.time()

        if args.local_rank == 0:  # Print them in the Process 0
            print('==================>Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                batch_idx, len(val_loader),
                args.world_size * opt.batch_size / batch_time.val,
                args.world_size * opt.batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses))

    if args.local_rank == 0:  # Print them in the Process 0
        # Write the log file each epoch.
        os.makedirs(checkpoint_path, exist_ok=True)
        logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
        logger.write('\tval_loss: {}'.format(losses.avg))  # validation鏃朵笉瑕乗n鎹㈣
        logger.flush()
        logger.close()


def adjust_learning_rate(optimizer, epoch, step, len_epoch, use_warmup=False):
    factor = epoch // 15

    if epoch >= 78:
        factor = (epoch - 78) // 5

    lr = opt.learning_rate * args.world_size * (0.2 ** factor)

    """Warmup"""
    if use_warmup:
        if epoch < 3:
            # print('=============>  Using warm-up learning rate....')
            lr = lr * float(1 + step + epoch * len_epoch) / (3. * len_epoch)  # len_epoch=len(train_loader)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cyclic(optimizer, current_epoch, start_epoch, swa_freqent=5, lr_max=4e-5, lr_min=2e-5):
    epoch = current_epoch - start_epoch

    lr = lr_max - (lr_max - lr_min) / (swa_freqent - 1) * (epoch - epoch // swa_freqent * swa_freqent)
    lr = round(lr, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor):
    # Reduces the tensor data across all machines
    # If we print the tensor, we can get:
    # tensor(334.4330, device='npu:1') *********************, here is npu:  npu:1
    # tensor(359.1895, device='npu:3') *********************, here is npu:  npu:3
    # tensor(263.3543, device='npu:2') *********************, here is npu:  npu:2
    # tensor(340.1970, device='npu:0') *********************, here is npu:  npu:0
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        test(epoch)
