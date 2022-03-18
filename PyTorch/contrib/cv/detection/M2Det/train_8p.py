#coding=utf-8
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
from __future__ import print_function
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import torch.nn as nn
import random
import shutil
import argparse
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *
from torch.nn.parallel import DistributedDataParallel as DDP
#amp淇敼寮�濮�
import apex
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from apex.optimizers import NpuFusedSGD
#amp淇敼缁撴潫

#NPU淇敼寮�濮�
import torch.npu
#NPU淇敼缁撴潫

parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det320_vgg16.py')
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus or npus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
#amp鍙傛暟寮�濮�
parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
parser.add_argument('--opt_level', type=str, default='O2')
parser.add_argument('--loss_scale', type=str, default=128.0)
#amp鍙傛暟缁撴潫
#distributed鍙傛暟寮�濮�
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--dist_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--multiprocessing_distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N NPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
##distributed鍙傛暟缁撴潫
#NPU鍙傛暟寮�濮�
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
parser.add_argument('--addr', default='127.0.0.1', type=str,
                        help='master addr')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cuda:0(device_id), npu:0(device_id).')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--FusedSGD', default=False, action='store_true',
                        help='use FusedSGD during trian')
#NPU鍙傛暟缁撴潫
parser.add_argument('--performance', default=False, action='store_true',
                        help='get performace of train model')
args = parser.parse_args()


#seed淇濊瘉閲嶅
def set_seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

# 璁剧疆hook func
def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        # 璇蜂緷鎹娇鐢ㄥ満鏅嚜瀹氫箟鍑芥暟
        print(name+' inputs', inputs)
        print(name+' outputs', outputs)
    return hook_function

def run(net, optimizer, criterion, images, priors, targets):
    out = net(images)
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, priors, targets)
    loss = loss_l + loss_c
    if args.amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = args.process_device_map[gpu]
    print('ngpus_per_node:{}'.format(ngpus_per_node))
    if args.dist_rank % ngpus_per_node == 0:
        logr = set_train_log()
        batch_time = AverageMeter('Time', ':6.3f', start_count_index=6)
        if args.performance:
            perf_time = time.time()
    
    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    if args.gpu is not None:
        if args.device.startswith('cuda'):
            print("[gpu id:",args.gpu,"]","Use GPU: {} for training".format(args.gpu))
        elif args.device.startswith('npu'):
            print("[npu id:",args.gpu,"]","Use NPU: {} for training".format(args.gpu))
        
    if args.distributed:
        if args.dist_url == "env://" and args.dist_rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.dist_rank = args.dist_rank * ngpus_per_node + gpu
        print("rank:",args.dist_rank)
        print("worksize",args.world_size)
        if args.device.startswith('npu'):
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                  world_size=args.world_size, rank=args.dist_rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                  world_size=args.world_size, rank=args.dist_rank)

    calculate_device = 'npu:{}'.format(gpu)#涓嶆槸args.gpu
    print("device",calculate_device)
    torch.npu.set_device(calculate_device)
    prof_kwargs = {'use_npu': True}#prof

    logger = set_logger(args.tensorboard)
    
    global cfg
    cfg = Config.fromfile(args.config)
    net = build_net('train', 
                    size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args.resume_net) # init the network with pretrained weights or resumed weights
    
    net = net.to(calculate_device)
    #cudnn.benchmark = True

    # 娉ㄥ唽姝ｅ弽鍚慼ook
    #for name, module in net.named_modules():
    #    module.register_forward_hook(hook_func('[forward]: '+name, module))
    #    module.register_backward_hook(hook_func('[backward]: '+name, module))
    #for name, param in net.named_parameters():
    #    print("[grad]: "+name, param.grad)

    #optimizer = set_optimizer(net, cfg)
    if args.FusedSGD:
        optimizer = NpuFusedSGD(
            net.parameters(),
            lr=cfg.train_cfg.lr[0],
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
            )
    else:
        optimizer = set_optimizer(net, cfg)
    
    #amp淇敼寮�濮�
    if args.amp:
        print("=> use amp, level is", args.opt_level)
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level=args.opt_level,
                                        loss_scale=args.loss_scale
                                        )
    #amp淇敼缁撴潫
    if args.distributed:
        #net = torch.nn.DataParallel(net)
        #鍒嗗彂妯″瀷
        net = DDP(net, device_ids=[args.gpu])
        
    
    criterion = set_criterion(cfg)
    priorbox = PriorBox(anchors(cfg))
    ############## npu modify begin #############
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(calculate_device)
    ############## npu modify end #############
    
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...',['yellow','bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    #鍒嗗彂鏁版嵁
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    print('epoch_size:{}'.format(epoch_size)) #鑷繁瀹氫箟
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    print('max_iter:{}'.format(max_iter)) #鑷繁瀹氫箟
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    print(stepvalues)
    print_info('===> Training M2Det on ' + args.dataset, ['yellow','bold'])
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    logr.info('start training!')
    for iteration in range(start_iter, max_iter):

        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, 
                                                  cfg.train_cfg.per_batch_size, 
                                                  shuffle=(train_sampler is None), 
                                                  num_workers=cfg.train_cfg.num_workers,
                                                  pin_memory=False,
                                                  sampler=train_sampler,
                                                  drop_last=True, 
                                                  collate_fn=detection_collate))
            #璁剧疆epoch浣嶇疆锛岃繖搴旇鏄釜涓轰簡鍚屾鎵�鍋氱殑宸ヤ綔
            if args.distributed:
                train_sampler.set_epoch(epoch)
                
            if epoch % cfg.model.save_eposhs == 0:
                save_checkpoint(net, cfg, final=False, datasetname = args.dataset, epoch=epoch)
            epoch += 1

        #torch.npu.synchronize()
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size, cfg)
        data_t0 = time.time()
        images, targets = next(batch_iterator)
        data_time = time.time()-data_t0
        ############## npu modify begin #############
        images = images.to(calculate_device)
        targets = [anno.to(calculate_device) for anno in targets]
        ############## npu modify end #############
        out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        write_logger({'loc_loss':loss_l.item(),
                      'conf_loss':loss_c.item(),
                      'loss':loss.item()},logger,iteration,status=args.tensorboard)
        #amp淇敼寮�濮�
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        #amp淇敼缁撴潫

        optimizer.step()
        

        #torch.npu.synchronize()        
        load_t1 = time.time()
        # measure elapsed time
        batch_time.update(load_t1 - load_t0)
        if iteration == 25:
            # 4. 鎵цforward+profiling
            with torch.autograd.profiler.profile(**prof_kwargs) as prof:
                run(net, optimizer, criterion, images, priors, targets)
            #print(prof.key_averages().table())
            prof.export_chrome_trace("pytorch_prof_%s.prof" % args.device)
            
        if args.dist_rank % ngpus_per_node == 0:
            print_train_log(iteration, cfg.train_cfg.print_epochs,
                            [time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),load_t1-load_t0,lr])
            if iteration % cfg.train_cfg.print_epochs == 0:
                logr.info('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||data_Time:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}'\
                .format(time.ctime(),epoch,iteration%epoch_size,epoch_size,iteration,loss_l.item(),loss_c.item(),data_time,load_t1-load_t0,lr))
            if (iteration+1) % epoch_size == 0:#鍒ゆ柇鏄惁瀹屾垚涓�涓猠poch
                if batch_time.avg > 0:
                    FPS = cfg.train_cfg.per_batch_size * args.ngpu / batch_time.avg
                    logr.info('Time:{}||Epoch:{}||Iter:{}||Batch_Time:{:.4f}||batch_time_avg:{:.3f}||FPS:{:.3f}'\
                              .format(time.ctime(),epoch,iteration,load_t1-load_t0,batch_time.avg,FPS))
                    print('Time:{}||Epoch:{}||Iter:{}||Batch_Time:{:.4f}||batch_time_avg:{:.3f}||Fps:{:.3f}'\
                              .format(time.ctime(),epoch,iteration,load_t1-load_t0,batch_time.avg,FPS))
                    batch_time.reset()
        if args.performance:
            perform_time = time.time() - perf_time
            #if perform_time > 1800 or iteration > 1000:
            if epoch > 2:
                break
    logr.info('finish training!')
    save_checkpoint(net, cfg, final=True, datasetname=args.dataset,epoch=-1)

if __name__ == '__main__':
    print_info('----------------------------------------------------------------------\n'
               '|                       M2Det Training Program                       |\n'
               '----------------------------------------------------------------------',['yellow','bold'])
    if args.seed is not None:
        set_seed_everything(args.seed)
    
    # 8p, 鎸囧畾璁粌鏈嶅姟鍣ㄧ殑ip鍜岀鍙�
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29501'
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    print('addr:{}'.format(args.addr))

    #args.distributed鍜宎rgs.multiprocessing_distributed鍔熻兘涓嶄竴鏍凤紝涓�涓槸鍒嗗竷鏁版嵁锛屼竴涓槸澶氱嚎绋嬪鐞�  
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed:
        if args.ngpu <= 1:
            print('the number of ngpu <= 1, distributed error!')
            
    # 8p锛屽垱寤虹敱device_id鍒皃rocess_id鐨勬槧灏勫弬鏁帮紝鑾峰彇鍗曡妭鐐规槆鑵�910 AI澶勭悊鍣ㄦ暟閲忋��
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    print('process_device_map:{}'.format(args.process_device_map))
    
    if args.device.startswith('npu'):
        ngpus_per_node = len(args.process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()
    
    print('{} node found.'.format(ngpus_per_node))
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)
    
