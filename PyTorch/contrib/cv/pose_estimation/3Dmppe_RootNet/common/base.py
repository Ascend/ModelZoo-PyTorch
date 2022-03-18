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
import os.path as osp
import math
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import cfg
from model import get_pose_net
from dataset import DatasetLoader
from multiple_datasets import MultipleDatasets
import apex
from apex.optimizers import NpuFusedAdam
from collections import OrderedDict
from torch.nn.parallel.data_parallel import DataParallel


# dynamic dataset import
for i in range(len(cfg.trainset_3d)):
    exec('from ' + cfg.trainset_3d[i] + ' import ' + cfg.trainset_3d[i])
for i in range(len(cfg.trainset_2d)):
    exec('from ' + cfg.trainset_2d[i] + ' import ' + cfg.trainset_2d[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar'), map_location=torch.device('cpu'))
        start_epoch = ckpt['epoch'] + 1

        # create new OrderedDict that does not contain `module.`
        state_dict = ckpt['network']
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
            new_state_dict = ckpt['network']
        model.load_state_dict(new_state_dict)

        optimizer.load_state_dict(ckpt['optimizer'])
        return start_epoch, model, optimizer


class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        optimizer = NpuFusedAdam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(DatasetLoader(eval(cfg.trainset_3d[i])("train"), True, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        )))

        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(DatasetLoader(eval(cfg.trainset_2d[i])("train"), True, transforms.Compose([\
                                                                                                        transforms.ToTensor(),
                                                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                                                        )))


        trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        
        
        
        if cfg.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(trainset_loader)
            self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.npus_per_node / cfg.batch_size)
            self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.batch_size, shuffle=False,
                                              num_workers=cfg.num_thread, pin_memory=False, sampler=self.train_sampler, drop_last=True) #NUP时pin_memory=False
            self.logger.info('distributed training.')
        else:
            self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.npus_per_node / cfg.batch_size)
            self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.batch_size, shuffle=True,
                                              num_workers=cfg.num_thread, pin_memory=False) #NUP时pin_memory=False
            self.logger.info('Undistributed training.')

    def _make_model(self, rank):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net(cfg, True)
        optimizer = self.get_optimizer(model)
        
        if cfg.continue_train:
            self.logger.info("Continue training mode.")
            start_epoch, model, optimizer = self.load_model(model, optimizer)
            self.logger.info("Model loading complete.")
        else:
            start_epoch = 0
        
        if cfg.distributed:
            local_rank = int(rank)
            torch.npu.set_device(local_rank)
            device = torch.device('npu', local_rank)
            os.environ['MASTER_ADDR'] = cfg.addr
            os.environ['MASTER_PORT'] = cfg.port
            dist.init_process_group(backend='hccl', #init_method="env://",
                                    world_size=cfg.world_size, rank=rank)
            model = model.to(device)
            if cfg.amp:
                model, optimizer = apex.amp.initialize(model, optimizer, opt_level=cfg.opt_level, loss_scale=cfg.loss_scale, combine_grad=False)
            model = DDP(model, device_ids=[rank], broadcast_buffers=False)
        else:
            if 'npu' in cfg.npu_device:
                torch.npu.set_device(cfg.npu_device)
            model = model.to(cfg.npu_device)
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=cfg.opt_level, loss_scale=cfg.loss_scale, combine_grad=False) 
            
        model.train()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):
    
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(cfg.testset)("test")
        testset_loader = DatasetLoader(testset, False, transforms.Compose([\
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]\
                                                                        ))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.test_batch_size, shuffle=False,
                                     num_workers=cfg.num_thread, pin_memory=False)
        
        self.testset = testset
        self.batch_generator = batch_generator
    
    def _make_model(self, npu_device_test):
        
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(cfg, False)
        model = model.to(npu_device_test)
        ckpt = torch.load(model_path)

        # create new OrderedDict that does not contain `module.`
        state_dict = ckpt['network']
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
            new_state_dict = ckpt['network']
        model.load_state_dict(new_state_dict)

        model.eval()
        self.model = model

    def _evaluate(self, preds, result_save_path):
        AP, result = self.testset.evaluate(preds, result_save_path)
        result = 'epoch:' + str(self.test_epoch) + ' ' + result
        self.logger.info(result)
        return AP, result
