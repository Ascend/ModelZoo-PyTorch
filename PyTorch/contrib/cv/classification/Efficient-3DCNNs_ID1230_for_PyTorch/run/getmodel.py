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
import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from models.mobilenetv2 import get_fine_tuning_parameters
from run.opts import parse_opts
from run.mean import get_mean, get_std
from run.spatial_transforms import *
from run.temporal_transforms import *
from run.target_transforms import ClassLabel, VideoID
from run.dataset import get_training_set, get_validation_set, get_test_set
from run.train import train_epoch
from run.utils import adjust_learning_rate, save_checkpoint, Logger, AverageMeter
from run.validation import val_epoch
from run.test import test
# import test
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil

from models import mobilenetv2

def generate_model(opt):
    assert opt.model in ['mobilenetv2']

    model = mobilenetv2.get_model(
        num_classes=opt.n_classes,
        sample_size=opt.sample_size,
        width_mult=opt.width_mult, opt=opt)

    return model


def change_model_pretrained(opt, model):
    if not opt.no_drive:
        model = nn.DataParallel(model, device_ids=None)
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.root_path + '/pretrain' + 'kinetics_mobilenetv2_1.0x_RGB_16_best_dp.pth'))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            model = model.module
            state = {
                'epoch': pretrain['epoch'],
                'arch': pretrain['arch'],
                'state_dict': model.state_dict(),
                'optimizer': pretrain['optimizer'],
                'best_prec1': pretrain['best_prec1']
            }

            torch.save(state, '%s/%s.pth' % (opt.root_path + '/pretrain', 'kinetics_mobilenetv2_1.0x_RGB_16_best'))
            print('change sucess')

def replacemudule(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict

def model_load_pretrained(opt, model):
    if not opt.pretrain_path and not opt.resume_path:
        print('complete...')
        opt.ft_portion = 'complete'

    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
        assert opt.arch == pretrain['arch']
        model.load_state_dict(replacemudule(pretrain['state_dict']))

        opt.ft_portion = 'last_layer'
        model.classifier = nn.Sequential(
                        nn.Dropout(opt.droupout_rate),
                        nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes)
                        )

    parameters = get_fine_tuning_parameters(model, opt.ft_portion)
    return model, parameters


