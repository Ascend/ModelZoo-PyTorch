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

from __future__ import print_function

import sys

sys.path.insert(0, '.')

import os
import torch
if torch.__version__ >= '1.8':
    import torch_npu
from torch.autograd import Variable
import torch.npu
import torch.nn as nn
import torch.distributed as dist
import apex
from apex import amp
import random
import time
import numpy as np
import argparse
from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss
from aligned_reid.utils.utils import str2bool
from aligned_reid.utils.utils import may_set_mode
from aligned_reid.utils.utils import load_ckpt
from aligned_reid.utils.utils import save_ckpt
from aligned_reid.utils.utils import set_devices
from aligned_reid.utils.utils import AverageMeter
from aligned_reid.utils.utils import to_scalar
from aligned_reid.utils.utils import set_seed
from aligned_reid.utils.utils import adjust_lr_exp
from aligned_reid.utils.utils import adjust_lr_staircase

MAX = 2147483647


def gen_seeds(num):
    return torch.randint(1, MAX, size=(num,), dtype=torch.float)


seed_init = 0


parser_ddp = argparse.ArgumentParser(description='PyTorch AlignedReID Training')
parser_ddp.add_argument('--data_pth', type=str, help='path to dataset')
parser_ddp.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
parser_ddp.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
parser_ddp.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
parser_ddp.add_argument('--dist-url', default='tcp://172.17.0.2:20987', type=str,
                        help='url used to set up distributed training')
parser_ddp.add_argument('--dist-backend', default='hccl', type=str,
                        help='distributed backend')
parser_ddp.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
parser_ddp.add_argument('--ids_per_batch', type=int, default=32)
parser_ddp.add_argument('--base_lr', type=float, default=2e-4)
parser_ddp.add_argument('--exp_decay_at_epoch', type=int, default=1000)
parser_ddp.add_argument('--total_epochs', type=int, default=300)
parser_ddp.add_argument('--only_test', type=str2bool, default=False)
parser_ddp.add_argument('--model_weight_file', type=str, default='')
parser_ddp.add_argument('--npu', default=0, type=int,
                        help='NPU id to use.')
parser_ddp.add_argument('--addr', default='172.17.0.2', type=str,
                        help='master addr')
parser_ddp.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser_ddp.add_argument('--seed', default=1234, type=int,
                        help='seed for initializing training. ')
parser_ddp.add_argument('--log_to_file', type=str2bool, default=False)



class Config(object):

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(2,))
        parser.add_argument('-r', '--run', type=int, default=1)
        parser.add_argument('--set_seed', type=str2bool, default=True)
        parser.add_argument('--dataset', type=str, default='market1501',
                            choices=['market1501', 'cuhk03', 'duke', 'combined'])
        parser.add_argument('--trainset_part', type=str, default='trainval',
                            choices=['trainval', 'train'])

        # Only for training set.
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--crop_prob', type=float, default=0)
        parser.add_argument('--crop_ratio', type=float, default=1)
        parser.add_argument('--ims_per_id', type=int, default=4)

        parser.add_argument('--normalize_feature', type=str2bool, default=False)
        parser.add_argument('--local_dist_own_hard_sample',
                            type=str2bool, default=True)
        parser.add_argument('-gm', '--global_margin', type=float, default=0.3)
        parser.add_argument('-lm', '--local_margin', type=float, default=0.3)
        parser.add_argument('-glw', '--g_loss_weight', type=float, default=1)
        parser.add_argument('-llw', '--l_loss_weight', type=float, default=0)
        parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0)

        parser.add_argument('--resume', type=str2bool, default=False)
        parser.add_argument('--exp_dir', type=str, default='')

        parser.add_argument('--lr_decay_type', type=str, default='exp',
                            choices=['exp', 'staircase'])

        parser.add_argument('--staircase_decay_at_epochs',
                            type=eval, default=(101, 201,))
        parser.add_argument('--staircase_decay_multiply_factor',
                            type=float, default=0.1)


        args = parser.parse_known_args()[0]
        args_ddp = parser_ddp.parse_args()
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        self.sys_device_n = len(self.sys_device_ids)

        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None

        # The experiments can be run for several times and performances be averaged.
        # `run` starts from `1`, not `0`.
        self.run = args.run

        ###########
        # Dataset #
        ###########

        # If you want to exactly reproduce the result in training, you have to set
        # num of threads to 1.
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 2

        self.dataset = args.dataset
        self.trainset_part = args.trainset_part

        # Image Processing

        # Just for training set
        self.crop_prob = args.crop_prob
        self.crop_ratio = args.crop_ratio
        self.resize_h_w = args.resize_h_w

        # Whether to scale by 1/255
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        self.ids_per_batch = args_ddp.ids_per_batch
        self.ims_per_id = args.ims_per_id
        self.train_final_batch = True
        self.train_mirror_type = ['random', 'always', None][0]
        self.train_shuffle = True

        self.test_batch_size = 32
        self.test_final_batch = True
        self.test_mirror_type = ['random', 'always', None][2]
        self.test_shuffle = False

        dataset_kwargs = dict(
            name=self.dataset,
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            num_prefetch_threads=self.prefetch_threads)

        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.train_set_kwargs = dict(
            part=self.trainset_part,
            ids_per_batch=self.ids_per_batch,
            ims_per_id=self.ims_per_id,
            final_batch=self.train_final_batch,
            shuffle=self.train_shuffle,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            mirror_type=self.train_mirror_type,
            prng=prng)
        self.train_set_kwargs.update(dataset_kwargs)

        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.test_set_kwargs = dict(
            part='test',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            prng=prng)
        self.test_set_kwargs.update(dataset_kwargs)

        ###############
        # ReID Model  #
        ###############

        self.local_dist_own_hard_sample = args.local_dist_own_hard_sample

        self.normalize_feature = args.normalize_feature

        self.local_conv_out_channels = 128
        self.global_margin = args.global_margin
        self.local_margin = args.local_margin

        # Identification Loss weight
        self.id_loss_weight = args.id_loss_weight

        # global loss weight
        self.g_loss_weight = args.g_loss_weight
        # local loss weight
        self.l_loss_weight = args.l_loss_weight

        #############
        # Training  #
        #############

        self.weight_decay = 0.0005

        # Initial learning rate
        self.base_lr = args_ddp.base_lr
        self.lr_decay_type = args.lr_decay_type
        self.exp_decay_at_epoch = args_ddp.exp_decay_at_epoch
        self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
        self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
        # Number of epochs to train
        self.total_epochs = args_ddp.total_epochs

        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.log_steps = 1

        # Only test and without training.
        self.only_test = args_ddp.only_test

        self.resume = args.resume

        #######
        # Log #
        #######

        self.log_to_file = args_ddp.log_to_file

        # Saving model weights and optimizer states, for resuming.
        self.ckpt_file = './ckpt.pth'
        # Just for loading a pretrained model; no optimizer states is needed.
        self.model_weight_file = args_ddp.model_weight_file


class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT

    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        global_feat = self.model(ims)
        global_feat = global_feat.data.cpu().numpy()
        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return global_feat


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map


def main_npu():
    args = parser_ddp.parse_args()

    if args.seed is not None:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '28889'

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    print(args.process_device_map)

    ngpus_per_node = len(args.process_device_map)

    print('{} node found.'.format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        # args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # print('mp.spawn')
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        # else:
        # Simply call main_worker function
        args.world_size = ngpus_per_node * args.world_size
        print("---------------args.npu", args.npu)
        main(args.npu, ngpus_per_node, args)


def main(npu, ngpus_per_node, args):

    cfg = Config()
    args.npu = args.process_device_map[npu]
    print('---------------args.npu', args.npu)

    CALCULATE_DEVICE = "npu:{}".format(args.npu)
    torch.npu.set_device(CALCULATE_DEVICE)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    TVT = set_devices(ngpus_per_node)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.npu
            print(args.rank)
            print(args.world_size)
            dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    if args.npu is 0:
        # Dump the configurations to log.
        import pprint
        print('&' * 120)
        print('-' * 60)
        print('cfg.__dict__')
        pprint.pprint(cfg.__dict__)
        print('-' * 60)

    ###########
    # Dataset #
    ###########
    train_set = create_dataset(device_id=args.npu, pth=args.data_pth, **cfg.train_set_kwargs)
    test_sets = []
    test_set_names = []
    if cfg.dataset == 'combined':
        for name in ['market1501', 'cuhk03', 'duke']:
            cfg.test_set_kwargs['name'] = name
            test_sets.append(create_dataset(**cfg.test_set_kwargs))
            test_set_names.append(name)
    else:
        test_sets.append(create_dataset(pth=args.data_pth, **cfg.test_set_kwargs))
        test_set_names.append(cfg.dataset)

    ###########
    # Models  #
    ###########

    model = Model(local_conv_out_channels=cfg.local_conv_out_channels,
                  num_classes=len(train_set.ids2labels))

    model = model.to(CALCULATE_DEVICE)
    # Model wrapper

    #############################
    # Criteria and Optimizers   #
    #############################

    id_criterion = nn.CrossEntropyLoss().to(CALCULATE_DEVICE)
    g_tri_loss = TripletLoss(margin=cfg.global_margin)
    l_tri_loss = TripletLoss(margin=cfg.local_margin)

    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(),
                                             lr=cfg.base_lr,
                                             weight_decay=cfg.weight_decay)

    # add apex 1
    amp.register_half_function(torch, 'addmm')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=128, combine_grad=True)
    modules_optims = [model, optimizer]

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.npu], broadcast_buffers=False,
                                                      find_unused_parameters=True)

    ################################
    # May Resume Models and Optims #
    ################################

    if cfg.resume:
        resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

    ########
    # Test #
    ########

    def test(load_model_weight=False):
        if load_model_weight:
            load_ckpt(modules_optims, cfg.model_weight_file)

        use_local_distance = False

        for test_set, name in zip(test_sets, test_set_names):
            test_set.set_feat_func(ExtractFeature(model, TVT))
            print('\n=========> Test on dataset: {} <=========\n'.format(name))
            test_set.eval(
                normalize_feat=cfg.normalize_feature,
                use_local_distance=use_local_distance)

    if cfg.only_test:
        if args.npu is 0:
            test(load_model_weight=True)
        return

    ############
    # Training #
    ############
    start_ep = resume_ep if cfg.resume else 0
    for ep in range(start_ep, cfg.total_epochs):

        # Adjust Learning Rate
        if cfg.lr_decay_type == 'exp':
            adjust_lr_exp(
                optimizer,
                cfg.base_lr,
                ep + 1,
                cfg.total_epochs,
                cfg.exp_decay_at_epoch)
        else:
            adjust_lr_staircase(
                optimizer,
                cfg.base_lr,
                ep + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)

        may_set_mode(modules_optims, 'train')

        g_prec_meter = AverageMeter()
        g_m_meter = AverageMeter()
        g_dist_ap_meter = AverageMeter()
        g_dist_an_meter = AverageMeter()
        g_loss_meter = AverageMeter()

        loss_meter = AverageMeter()

        ep_st = time.time()
        step = 0
        fps_all = 0
        epoch_done = False

        while not epoch_done:


            step += 1
            step_st = time.time()

            ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()
            ims_var = torch.from_numpy(ims).float()
            ims_var = ims_var.to(CALCULATE_DEVICE)
            labels_t = torch.from_numpy(labels).long()
            labels_t = labels_t.to(CALCULATE_DEVICE)
            labels_var = Variable(labels_t)

            global_feat = model(ims_var)
            local_feat = logits = global_feat
            g_loss, p_inds, n_inds, g_dist_ap, g_dist_an, g_dist_mat = global_loss(
                g_tri_loss, global_feat, labels_t,
                normalize_feature=cfg.normalize_feature)

            if cfg.l_loss_weight == 0:
                l_loss = 0
            elif cfg.local_dist_own_hard_sample:
                # Let local distance find its own hard samples.
                l_loss, l_dist_ap, l_dist_an, _ = local_loss(
                    l_tri_loss, local_feat, None, None, labels_t,
                    normalize_feature=cfg.normalize_feature)
            else:
                l_loss, l_dist_ap, l_dist_an = local_loss(
                    l_tri_loss, local_feat, p_inds, n_inds, labels_t,
                    normalize_feature=cfg.normalize_feature)

            id_loss = 0
            if cfg.id_loss_weight > 0:
                id_loss = id_criterion(logits, labels_var)

            loss = g_loss * cfg.g_loss_weight \
                   + l_loss * cfg.l_loss_weight \
                   + id_loss * cfg.id_loss_weight

            optimizer.zero_grad()
            # add apex 2
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            ############
            # Step Log #
            ############

            # precision
            g_prec = (g_dist_an > g_dist_ap).data.float().mean()
            # the proportion of triplets that satisfy margin
            g_m = (g_dist_an > g_dist_ap + cfg.global_margin).data.float().mean()
            g_d_ap = g_dist_ap.data.mean()
            g_d_an = g_dist_an.data.mean()

            g_prec_meter.update(g_prec)
            g_m_meter.update(g_m)
            g_dist_ap_meter.update(g_d_ap)
            g_dist_an_meter.update(g_d_an)
            g_loss_meter.update(to_scalar(g_loss))

            loss_meter.update(to_scalar(loss))

            if args.npu is 0:
                if step % cfg.log_steps == 0:
                    time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
                        step, ep + 1, time.time() - step_st, )

                    if cfg.g_loss_weight > 0:
                        if step > 3:
                            fps_step = len(args.process_device_map) * cfg.ids_per_batch / (time.time() - step_st)
                            fps_all += fps_step
                            g_log = (', gp {:.2%}, gm {:.2%}, '
                                     'gd_ap {:.4f}, gd_an {:.4f}, '
                                     'gL {:.4f}, fps {:.2f}'.format(
                                g_prec_meter.val, g_m_meter.val,
                                g_dist_ap_meter.val, g_dist_an_meter.val,
                                g_loss_meter.val, fps_step))
                        else:
                            g_log = (', gp {:.2%}, gm {:.2%}, '
                                     'gd_ap {:.4f}, gd_an {:.4f}, '
                                     'gL {:.4f}'.format(
                                g_prec_meter.val, g_m_meter.val,
                                g_dist_ap_meter.val, g_dist_an_meter.val,
                                g_loss_meter.val, ))
                    else:
                        g_log = ''
                    l_log = ''
                    id_log = ''
                    total_loss_log = ', loss {:.4f}'.format(loss_meter.val)

                    log = time_log + \
                          g_log + l_log + id_log + \
                          total_loss_log
                    print(log)

        #############
        # Epoch Log #
        #############
        if args.npu is 0:
            FPS = fps_all / (step - 3)
            time_log = 'Ep {}, {:.2f}s, FPS {:.2f}'.format(ep + 1, time.time() - ep_st, fps_step)

            if cfg.g_loss_weight > 0:
                g_log = (', gp {:.2%}, gm {:.2%}, '
                         'gd_ap {:.4f}, gd_an {:.4f}, '
                         'gL {:.4f}'.format(
                    g_prec_meter.avg, g_m_meter.avg,
                    g_dist_ap_meter.avg, g_dist_an_meter.avg,
                    g_loss_meter.avg, ))
            else:
                g_log = ''
            l_log = ''
            id_log = ''
            total_loss_log = ', loss {:.4f}'.format(loss_meter.avg)

            log = time_log + \
                  g_log + l_log + id_log + \
                  total_loss_log
            print(log)

        # save ckpt
        if cfg.log_to_file:
            if args.npu is 0:
                save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

    ########
    # Test #
    ########
    if args.npu is 0:
        test(load_model_weight=False)


if __name__ == '__main__':
    main_npu()
