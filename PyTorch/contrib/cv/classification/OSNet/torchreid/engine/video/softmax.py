"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import division, print_function, absolute_import
import torch

from torchreid.engine.image import ImageSoftmaxEngine


class VideoSoftmaxEngine(ImageSoftmaxEngine):
    """Softmax-loss engine for video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        pooling_method (str, optional): how to pool features for a tracklet.
            Default is "avg" (average). Choices are ["avg", "max"].

    Examples::
        
        import torch
        import torchreid
        # Each batch contains batch_size*seq_len images
        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            combineall=False,
            batch_size=8, # number of tracklets
            seq_len=15 # number of images in each tracklet
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.VideoSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler,
            pooling_method='avg'
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-mars',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        pooling_method='avg'
    ):
        super(VideoSoftmaxEngine, self).__init__(
            datamanager,
            model,
            optimizer,
            scheduler=scheduler,
            use_gpu=use_gpu,
            label_smooth=label_smooth
        )
        self.pooling_method = pooling_method

    def parse_data_for_train(self, data):
        imgs = data['img']
        pids = data['pid']
        if imgs.dim() == 5:
            # b: batch size
            # s: sqeuence length
            # c: channel depth
            # h: height
            # w: width
            b, s, c, h, w = imgs.size()
            imgs = imgs.view(b * s, c, h, w)
            pids = pids.view(b, 1).expand(b, s)
            pids = pids.contiguous().view(b * s)
        return imgs, pids

    def extract_features(self, input):
        # b: batch size
        # s: sqeuence length
        # c: channel depth
        # h: height
        # w: width
        b, s, c, h, w = input.size()
        input = input.view(b * s, c, h, w)
        features = self.model(input)
        features = features.view(b, s, -1)
        if self.pooling_method == 'avg':
            features = torch.mean(features, 1)
        else:
            features = torch.max(features, 1)[0]
        return features
