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

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        loc = 'npu:{}'.format(self.args.local_rank)
        #self.device = torch.device(args.device)
        if args.device == 'npu':
            self.device = torch.npu.set_device(loc)
        else:
            self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.BatchNorm2d#nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(loc)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(loc)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        loc = 'npu:{}'.format(self.args.local_rank)
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(loc)
            target = target.to(loc)
            target = target.to(torch.int32)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                mask = get_color_pallete(predict, self.args.dataset)
                mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.device == "npu":
        args.device = "npu"
    elif not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.npu.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if args.model == "enet":
            outdir = '../runs/pred_pic/{}_{}'.format(args.model, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    log_filename = '{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset)
    if args.model == "enet":
        log_filename = '{}_{}_log.txt'.format(args.model, args.dataset)
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename=log_filename, mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.npu.empty_cache()
