#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return model

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args).to(device)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []),
            **args.get('loss_kwargs', {})).build().to(device)
        self.device = device

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred
