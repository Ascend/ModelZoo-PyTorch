# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import os
import time
import datetime
import yaml
import sys
import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.distributed as dist

from dataset import CityscapesDataset
from models import ICNet
from utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
import apex
from apex import amp
import numpy as np

import torch.onnx
import moxing as mox

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        is_distributed = cfg["apex"]["is_distributed"]
        if is_distributed == 0:
            self.device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
            torch.npu.set_device(self.device)
            workers = 4
            rank_id = 0
            self.rank_id = rank_id
            logger.info("workers:{}".format(workers))
        else:
            # distributed
            host_ip = sys.argv[4]
            logger.info("host_ip:{}".format(host_ip))
            os.environ['MASTER_ADDR'] = host_ip
            os.environ['MASTER_PORT'] = '29688'

            rank_id = int(sys.argv[5])
            self.rank_id = rank_id
            print('rank_id:', type(rank_id), rank_id)

            workers = int(sys.argv[6])
            logger.info("workers:{}".format(workers))

            loc = 'npu:{}'.format(rank_id)
            self.device = torch.device(loc)
            torch.npu.set_device(loc)
            self.world_size = cfg["apex"]["world_size"]
            dist.init_process_group(backend=cfg["apex"]["dist_backend"], world_size=cfg["apex"]["world_size"],
                                    rank=rank_id)
            print(self.rank_id, cfg["apex"]["world_size"], 'dist.init_process_group done.')

        self.dataparallel = is_distributed

        # dataset and dataloader
        data_path = sys.argv[1]
        real_path = '/cache/data_url'
        if not os.path.exists(real_path):
            os.makedirs(real_path)
        mox.file.copy_parallel(data_path, real_path)
        print("training data finish copy to %s." % real_path)

        train_dataset = CityscapesDataset(root=real_path,
                                          split='train',
                                          base_size=cfg["model"]["base_size"],
                                          crop_size=cfg["model"]["crop_size"])
        val_dataset = CityscapesDataset(root=real_path,
                                        split='val',
                                        base_size=cfg["model"]["base_size"],
                                        crop_size=cfg["model"]["crop_size"])
        if is_distributed == 0:
            self.train_sampler = None
        else:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        self.train_dataloader = data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg["train"]["train_batch_size"],
                                                shuffle=(self.train_sampler is None),
                                                num_workers=workers,
                                                pin_memory=False,
                                                sampler=self.train_sampler,
                                                drop_last=False)
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=False,
                                              drop_last=False)

        self.iters_per_epoch = len(self.train_dataloader)
        self.max_iters = cfg["train"]["epochs"] * self.iters_per_epoch

        # create network

        if len(sys.argv) == 6 or len(sys.argv) == 9:
            self.model = ICNet(nclass=train_dataset.NUM_CLASS, backbone='resnet50')
            if sys.argv[5] == '1':
                cpt = torch.load(sys.argv[4], map_location='cpu')
                cpt.pop('head.cff_12.conv_low_cls.weight')
                cpt.pop('head.cff_24.conv_low_cls.weight')
                cpt.pop('head.conv_cls.weight')
                self.model.load_state_dict(cpt, strict=False)
            else:
                cpt = torch.load(sys.argv[8], map_location='cpu')
                cpt.pop('head.cff_12.conv_low_cls.weight')
                cpt.pop('head.cff_24.conv_low_cls.weight')
                cpt.pop('head.conv_cls.weight')
                self.model.load_state_dict(cpt, strict=False)
        else:
            self.model = ICNet(nclass=train_dataset.NUM_CLASS, backbone='resnet50').to(self.device)

        self.model = self.model.to(self.device)  # model to device

        # create criterion
        self.criterion = ICNetLoss(ignore_index=train_dataset.IGNORE_INDEX).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': cfg["optimizer"]["init_lr"]})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append(
                    {'params': getattr(self.model, module).parameters(), 'lr': cfg["optimizer"]["init_lr"] * 10})
        self.base_lr = cfg["optimizer"]["init_lr"]
        self.optimizer = apex.optimizers.NpuFusedSGD(params=params_list,
                                                     lr=cfg["optimizer"]["init_lr"],
                                                     momentum=cfg["optimizer"]["momentum"],
                                                     weight_decay=cfg["optimizer"]["weight_decay"])

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", loss_scale=32.0)

        # lr scheduler
        self.lr_scheduler = IterationPolyLR(self.optimizer,
                                            max_iters=self.max_iters,
                                            power=0.9)
        if (self.dataparallel):
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank_id])
            print(self.rank_id, 'init ddp done.')

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.NUM_CLASS)

        self.current_mIoU = 0.0
        self.best_mIoU = 0.0

        self.epochs = int(sys.argv[2])
        self.current_epoch = 0
        self.current_iteration = 0

    @staticmethod
    def adjust_learning_rate(optimizer, base_lr, epoch, epochs, warm_up_epochs=3):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if warm_up_epochs > 0 and epoch < warm_up_epochs:
            lr = base_lr * ((epoch + 1) / (warm_up_epochs + 1))
        else:
            alpha = 0
            cosine_decay = 0.5 * (
                    1 + np.cos(np.pi * (epoch - warm_up_epochs) / (epochs - warm_up_epochs)))
            decayed = (1 - alpha) * cosine_decay + alpha
            lr = base_lr * decayed

        print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        batch_time = AverageMeter('Time', ':6.3f')

        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        num_devices = len(cfg["train"]["specific_npu_num"].split(','))

        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()

        for epoch in range(self.epochs):
            if self.dataparallel:
                self.train_sampler.set_epoch(epoch)

            self.adjust_learning_rate(self.optimizer, self.base_lr, epoch, epochs)

            self.current_epoch += 1
            list_loss = []
            self.metric.reset()
            end = time.time()
            train_batch_size = self.cfg["train"]["train_batch_size"]
            for i, (images, targets, _) in enumerate(self.train_dataloader):
                self.current_iteration += 1
                images = images.to(self.device)
                targets = targets.to(torch.int32).to(self.device)
                if self.dataparallel == 0:
                    # get op list
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                        list_loss.append(loss.item())

                        self.optimizer.zero_grad()
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                        self.optimizer.step()
                    prof.export_chrome_trace("output.prof")
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    list_loss.append(loss.item())

                    self.optimizer.zero_grad()
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    self.optimizer.step()

                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (
                            max_iters - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_time.avg > 0:
                    FPS = train_batch_size * num_devices / batch_time.avg
                else:
                    FPS = 0

                if self.current_iteration % log_per_iters == 0 and self.rank_id == 0:
                    logger.info(
                        "RankID[{}] Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}, FPS: {:.3f}".format(
                            self.rank_id,
                            self.current_epoch, self.epochs,
                            self.current_iteration, max_iters,
                            self.optimizer.param_groups[0]['lr'],
                            loss.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string,
                            FPS))
            average_loss = sum(list_loss) / len(list_loss)
            if self.rank_id == 0:
                logger.info(
                    "RankID[{}] Epochs: {:d}/{:d}, Average loss: {:.3f}, FPS: {:.3f}".format(self.rank_id,
                                                                                             self.current_epoch,
                                                                                             self.epochs, average_loss,
                                                                                             FPS))

            if (self.current_iteration % val_per_iters == 0) or (epoch > int(0.9 * self.epochs)):
                self.validation()
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        if self.rank_id == 0:
            logger.info(
                "RankID[{}] Total training time: {} ({:.4f}s / it)".format(
                    self.rank_id, total_training_str, total_training_time / max_iters))

    def validation(self):
        batch_time = AverageMeter('Time', ':6.3f')

        is_best = False
        self.metric.reset()
        if self.dataparallel:
            model = self.model.module
        else:
            model = self.model
        model.eval()
        lsit_pixAcc = []
        list_mIoU = []
        list_loss = []
        end = time.time()
        valid_batch_size = self.cfg["train"]["valid_batch_size"]
        num_devices = len(cfg["train"]["specific_npu_num"].split(','))
        for i, (image, targets, filename) in enumerate(self.val_dataloader):

            image = image.to(self.device)
            targets = targets.to(torch.int32).to(self.device)

            with torch.no_grad():
                outputs = model(image)
                loss = self.criterion(outputs, targets)
            self.metric.update(outputs[0], targets)
            pixAcc, mIoU = self.metric.get()
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            list_loss.append(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_time.avg > 0:
                FPS = valid_batch_size * num_devices / batch_time.avg
            else:
                FPS = 0

        average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        average_loss = sum(list_loss) / len(list_loss)
        self.current_mIoU = average_mIoU

        logger.info(
            "RankID[{}] Validation: Average loss: {:.3f}, Average mIoU: {:.3f}, Average pixAcc: {:.3f}, FPS: {:.3f}".format(
                self.rank_id, average_loss, average_mIoU, average_pixAcc, FPS))

        if self.current_mIoU > self.best_mIoU:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best and self.rank_id == 0:
            save_checkpoint(self.model, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.dataparallel)


def save_checkpoint(model, cfg, epoch=0, is_best=False, mIoU=0.0, dataparallel=False):
    """Save Checkpoint"""
    directory = 'cache/training'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{:.3f}.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch, mIoU)
    filename = os.path.join(directory, filename)
    if dataparallel:
        model = model.module
    if is_best:
        best_filename = '{}_{}_{}_{:.3f}_best_model.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"], epoch,
                                                                mIoU)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)
        onnx_file = os.path.join(directory, 'ICNet.onnx')
        convert(best_filename, onnx_file)
        mox.file.copy_parallel(directory, sys.argv[4])


def convert(pth_file, onnx_file):
    model = ICNet(nclass=19, backbone='resnet50', train_mode=False)
    pretrained_net = torch.load(pth_file, map_location='cpu')
    model.load_state_dict(pretrained_net)
    model.eval()
    input_names = ["actual_input_1"]
    with amp.disable_casts():
        dummy_input = torch.randn(1, 3, 1024, 2048)
        torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, opset_version=11, verbose=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == '__main__':
    # Set config file
    config_path = sys.argv[3]
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)

    # Use specific NPU
    os.environ["NPU_VISIBLE_DEVICES"] = str(cfg["train"]["specific_npu_num"])
    num_npus = len(cfg["train"]["specific_npu_num"].split(','))
    print("torch.npu.is_available(): {}".format(torch.npu.is_available()))
    print("torch.npu.device_count(): {}".format(torch.npu.device_count()))
    print("torch.npu.current_device(): {}".format(torch.npu.current_device()))

    # Set logger
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))
    logger.info("Using {} NPUs".format(num_npus))
    logger.info("torch.npu.is_available(): {}".format(torch.npu.is_available()))
    logger.info("torch.npu.device_count(): {}".format(torch.npu.device_count()))
    logger.info("torch.npu.current_device(): {}".format(torch.npu.current_device()))
    logger.info(cfg)

    # Start train
    trainer = Trainer(cfg)
    trainer.train()

