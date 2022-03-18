# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from tqdm import tqdm
from collections import OrderedDict
from apex import amp
import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from apex.optimizers import NpuFusedAdam

from dataloader import *
from model_RawNet2 import RawNet2
from parser1 import get_args
from trainer import *
from utils import *

import torch.npu

def main():
    # parse arguments
    args = get_args()
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '50000'

    # make experiment reproducible if specified
    if args.reproducible:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # get the account of the npu
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    process_device_map = device_id_to_process_device_map(args.device_list)
    if args.device == 'npu':
        ngpus_per_node = len(process_device_map)
    else:
        ngpus_per_node = torch.cuda.device_count()

    # get the param of local_rank
    local_rank = int(args.local_rank)
    torch.npu.set_device(local_rank)
    # dist.barrier()
    device = torch.device('npu', local_rank)
    args.bs = int(args.bs / ngpus_per_node)
    args.nb_worker = int((args.nb_worker + ngpus_per_node - 1) / ngpus_per_node)

    # initialize the process group
    if args.device == 'npu':
        dist.init_process_group(backend='hccl',  # init_method=args.dist_url,
                                world_size=args.world_size, rank=local_rank)
    else:
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=local_rank)

    #get utt_lists & define labels
    l_dev = sorted(get_utt_list(args.DB_vox2 + args.dev_wav))
    l_val = sorted(get_utt_list(args.DB + args.val_wav))
    l_eval = sorted(get_utt_list(args.DB + args.eval_wav))
    d_label_vox2 = get_label_dic_Voxceleb(l_dev)
    args.model['nb_classes'] = len(list(d_label_vox2.keys()))

    #def make_validation_trial(l_utt, nb_trial, dir_val_trial)
    if bool(False):
        make_validation_trial(l_utt = l_val, nb_trial = args.nb_val_trial, dir_val_trial = args.DB + 'val_trial.txt')
    with open(args.DB + 'val_trial.txt', 'r') as f:
        l_val_trial = f.readlines()
    with open(args.DB + 'veri_test.txt', 'r') as f:
        l_eval_trial = f.readlines()

    #define dataset generators
    devset = Dataset_VoxCeleb2(list_IDs = l_dev,
         labels = d_label_vox2,
         nb_samp = args.nb_samp,
         base_dir = args.DB_vox2 + args.dev_wav)
    # add distributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(devset)
    # suit the load data
    devset_gen = torch.utils.data.DataLoader(devset,
         batch_size = args.bs,
         shuffle = (train_sampler is None),
         pin_memory = False,
         drop_last = True,
         num_workers = args.nb_worker,
         sampler = train_sampler
         )
    valset = Dataset_VoxCeleb2(list_IDs = l_val,
         return_label = False,
         nb_samp = args.nb_samp,
         base_dir = args.DB)
    valset_gen = data.DataLoader(valset,
         batch_size = args.bs,
         shuffle = False,
         drop_last = False,
         num_workers = args.nb_worker)
    TA_evalset = TA_Dataset_VoxCeleb2(list_IDs = l_eval,
        return_label = False,
        window_size = args.window_size, # 20% of nb_samp
        nb_samp = args.nb_samp,
        base_dir = args.DB + args.eval_wav)
    TA_evalset_gen = torch.utils.data.DataLoader(TA_evalset,
        batch_size = 1,
        shuffle = False,
        drop_last = False,
        num_workers = args.nb_worker)

    #set save directory
    save_dir = args.save_dir + args.name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'results/'):
        os.makedirs(save_dir+'results/')
    if not os.path.exists(save_dir+'models/'):
        os.makedirs(save_dir+'models/')
    if not os.path.exists(save_dir + 'prof/'):
        os.makedirs(save_dir + 'prof/')
    if not os.path.exists(save_dir + 'log/'):
        os.makedirs(save_dir + 'log/')
        
    #log experiment parameters to local and comet_ml server
    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in sorted(vars(args).items()):
        print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    for k, v in sorted(args.model.items()):
        print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()



    # define model
    if bool(args.mg):
        #create an instance of NN
        model = RawNet2(args.model)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    else:
        model = RawNet2(args.model).to(device)
        if args.load_model: model.load_state_dict(torch.load(args.load_model_dir))
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    if not args.load_model:
        model.apply(init_weights)

    # move the cce to npu
    criterion = {}
    criterion['cce'] = nn.CrossEntropyLoss().to(device)

    # set optimizer
    params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if 'bn' not in name
            ]
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if 'bn' in name
            ],
            'weight_decay':
                0
        },
    ]


    if args.optimizer.lower() == 'sgd':
        model = DDP(model, device_ids = [local_rank], output_device = local_rank)
        if not args.load_model:
            optimizer = NpuFusedSgd(params,
                                    lr=args.lr,
                                    momentum=args.opt_mom,
                                    weight_decay=args.wd,
                                    nesterov=args.nesterov)
        model = model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O2", loss_scale = 128.0)
        model = DDP(model, device_ids = [local_rank], broadcast_buffers=False)
    elif args.optimizer.lower() == 'adam':
        optimizer = NpuFusedAdam(params,
                                lr=args.lr,
                                weight_decay=args.wd,
                                amsgrad=args.amsgrad)
        # continue training
        if args.load_model:
            ckpt = torch.load(args.load_model_dir, map_location=torch.device('cpu'))
            state_dict = ckpt['model']
            remove_module = False
            for k, v in state_dict.items():
                if 'module.' in k:
                    remove_module = True
                    break
            if remove_module:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
            else:
                new_state_dict = ckpt['model']
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(ckpt['optimizer'])
            print("Load Model Successfully！")
            
        model = model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O2", loss_scale = 128.0)
        model = DDP(model, device_ids = [local_rank], broadcast_buffers=False)
    else:
        raise NotImplementedError('Add other optimizers if needed')
    #set learning rate decay
    if bool(args.do_lr_decay):
        if args.lr_decay == 'keras':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: keras_lr_decay(step))
        elif args.lr_decay == 'cosine':
            raise NotImplementedError('Not implemented yet')
        else:
            raise NotImplementedError('Not implemented yet')
     
    ##########################################
    #Train####################################
    ##########################################
    best_TA_eval_eer = 99.
    f_eer = open(save_dir + 'eers.txt', 'a', buffering = 1)
    for epoch in tqdm(range(args.epoch)):
        # train_sampler.set_epcoh(epoch)

        # train phase
        train_model(model = model,
            db_gen = devset_gen,
            args = args,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            criterion = criterion,
            device = device,
            epoch = epoch)
        
        # evaluate model
        TA_eval_eer = time_augmented_evaluate_model(mode='eval',
                                                    model=model,
                                                    db_gen=TA_evalset_gen,
                                                    l_utt=l_eval,
                                                    save_dir=save_dir,
                                                    epoch=epoch,
                                                    l_trial=l_eval_trial,
                                                    args=args,
                                                    device=device)
        f_eer.write('epoch:%d, TA_eval_eer:%.4f\n' % (epoch, TA_eval_eer))
        save_model_dict = model.state_dict()
        if float(TA_eval_eer) < best_TA_eval_eer:
            print('New best TA_EER: %f'%float(TA_eval_eer))
            best_TA_eval_eer = float(TA_eval_eer)
        torch.save({'epoch': epoch,
                        'model': save_model_dict,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()}, save_dir + 'models/TA_%d_%.4f.pt'%(epoch, TA_eval_eer))

    f_eer.close()


# set the specific device to train
def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
