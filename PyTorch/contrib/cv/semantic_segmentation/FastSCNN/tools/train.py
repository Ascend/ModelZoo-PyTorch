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
import copy
import datetime
import time
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import apex.amp as amp
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg

class Trainer(object):
    def __init__(self, args):
        self.args = args
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode=cfg.DATASET.MODE, **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create network
        if args.distributed:
            local_rank = torch.distributed.get_rank()
        else:
            torch.npu.set_device('npu:3')
            local_rank = 3
        self.device = f'npu:{local_rank}'
        args.device = f'npu:{local_rank}'
        self.model = get_segmentation_model().to(self.device)
        
        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)
       # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", loss_scale=128)
        
       # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)
       # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume,map_location=torch.device(self.device))
        

            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)
        # print params and flops
        # if get_rank() == 0:
        #     try:
        #          show_flops_params(copy.deepcopy(self.model), args.device)
        #     except Exception as e:
        #         logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info(' effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

     

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0


    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1
            logging.info("iteration=====>{}".format(iteration))
            if (iteration % iters_per_epoch) == 6:
                time_step5 = time.time()
            if iteration == 6:
                with torch.autograd.profiler.profile() as prof:
                    # self.device = f'npu:{local_rank}'
                    images = images.npu()
                    targets = targets.to(torch.int32).npu()
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    # reduce losses over all GPUs for logging purposes
                    # loss_dict_reduced = reduce_loss_dict(loss_dict)
                    # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    # train_loss += losses
                    self.optimizer.zero_grad()
                    with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    self.optimizer.step()
                prof.export_chrome_trace("output.prof")
                self.lr_scheduler.step()
            else:
                # self.device = f'npu:{local_rank}'
                images = images.npu()
                targets = targets.to(torch.int32).npu()
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())
                # # reduce losses over all GPUs for logging purposes
                # loss_dict_reduced = reduce_loss_dict(loss_dict)
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                self.optimizer.zero_grad()
                with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

            if (iteration % iters_per_epoch) == 0:
                time_epoch = time.time()
                time_avg = time_epoch - time_step5
                fps = iters_per_epoch*cfg.TRAIN.BATCH_SIZE * self.args.num_gpus / time_avg
                logging.info(
                    "num_gpus: {} || batch_size {}|| time_avg {}  ||FPS {}".format(self.args.num_gpus,
                                                                                   cfg.TRAIN.BATCH_SIZE,
                                                                                time_avg, fps))
            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # if iteration % log_per_iters == 0 and self.save_to_disk:
            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))
                torch.save(self.model.state_dict(), 'best_model.pth')
            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:

                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)
            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(epoch)
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, epoch):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        # torch.npu.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.npu()
            target = target.to(torch.int32).npu()
            with torch.no_grad():
                output = model(image)
                # if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                #     output = model(image)[0]
                # else:
                #     size = image.size()[2:]
                #     pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
                #     pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
                #     image = F.pad(image, (0, pad_height, 0, pad_width))
                #     output = model(image)[0]
                #     output = output[..., :size[0], :size[1]]

            self.metric.update(output[0], target)
            pixAcc, mIoU = self.metric.get()
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))
        pixAcc, mIoU = self.metric.get()
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        synchronize()
        if self.best_pred < mIoU and self.save_to_disk:
            self.best_pred = mIoU
            logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, save the model..'.format(epoch, pixAcc * 100, mIoU * 100))
            save_checkpoint(model, epoch, is_best=True)


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    trainer.train()
