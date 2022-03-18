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
# limitations under the License.from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.npu
from torch.nn.parallel import DataParallel

import apex
from apex import amp

import numpy as np
import argparse

from aligned_reid.dataset import create_dataset
from aligned_reid.model.Model import Model
from aligned_reid.utils.utilsn1 import str2bool
from aligned_reid.utils.utilsn1 import load_ckpt
from aligned_reid.utils.utilsn1 import set_devices
from aligned_reid.utils.utilsn1 import set_seed


parser_ddp = argparse.ArgumentParser(description='PyTorch AlignedReID Training')
parser_ddp.add_argument('--data_pth', type=str, default='./market1501/')
parser_ddp.add_argument('--pkl', type=str, default='')
parser_ddp.add_argument('--ids_per_batch', type=int, default=32)
parser_ddp.add_argument('--base_lr', type=float, default=2e-4)
parser_ddp.add_argument('--exp_decay_at_epoch', type=int, default=1000)
parser_ddp.add_argument('--total_epochs', type=int, default=300)
parser_ddp.add_argument('--model_weight_file', type=str, default='ckpt.pth')
parser_ddp.add_argument('--only_test', type=str2bool, default=True)


class Config(object):

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
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

        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('-gm', '--global_margin', type=float, default=0.3)
        parser.add_argument('-lm', '--local_margin', type=float, default=0.3)
        parser.add_argument('-glw', '--g_loss_weight', type=float, default=1)
        parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.)
        parser.add_argument('-idlw', '--id_loss_weight', type=float, default=0.)

        args = parser.parse_known_args()[0]
        args_ddp = parser_ddp.parse_args()

        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        self.sys_device_n = len(self.sys_device_ids)

        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None

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
        self.local_conv_out_channels = 128

        # Only test and without training.
        self.only_test = args_ddp.only_test

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
        self.model.train(old_train_eval_model)
        return global_feat


def main():
    cfg = Config()
    args = parser_ddp.parse_args()

    TVT, TMO = set_devices(cfg.sys_device_ids)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    ###########
    # Dataset #
    ###########
    train_set = create_dataset(pth=args.data_pth, pkl=args.pkl, **cfg.train_set_kwargs)

    test_sets = []
    test_set_names = []
    if cfg.dataset == 'combined':
        for name in ['market1501', 'cuhk03', 'duke']:
            cfg.test_set_kwargs['name'] = name
            test_sets.append(create_dataset(**cfg.test_set_kwargs))
            test_set_names.append(name)
    else:
        test_sets.append(create_dataset(pth=args.data_pth, pkl=args.pkl, **cfg.test_set_kwargs))
        test_set_names.append(cfg.dataset)

    ###########
    # Models  #
    ###########

    model = Model(local_conv_out_channels=cfg.local_conv_out_channels,
                  num_classes=len(train_set.ids2labels))

    model = model.npu()

    #############################
    # Criteria and Optimizers   #
    #############################

    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(),
                                             lr=cfg.base_lr,
                                             weight_decay=cfg.weight_decay)
    torch.backends.cudnn.enabled = True
    # add apex 1
    amp.register_half_function(torch, 'addmm')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=128, combine_grad=True)
    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]

    # Model wrapper
    model_w = DataParallel(model)
    model_w = model_w.npu()

    # May Transfer Models and Optims to Specified Device. Transferring optimizer
    # is to cope with the case when you load the checkpoint to a new device.
    TMO(modules_optims)

    def output(load_model_weight=False):
        if load_model_weight:
                load_ckpt(modules_optims, cfg.model_weight_file)
        print("=================Extract One Global Feat=================")
        for test_set, name in zip(test_sets, test_set_names):
            test_set.set_feat_func(ExtractFeature(model_w, TVT))
            test_set.eval(
                normalize_feat=cfg.normalize_feature,
                use_local_distance=False,
                feature_only=True)


    output(load_model_weight=True)


if __name__ == '__main__':
    main()
