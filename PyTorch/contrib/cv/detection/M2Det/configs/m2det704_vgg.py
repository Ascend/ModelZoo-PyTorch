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

model = dict(
    type = 'm2det',
    input_size = 704,
    init_net = True,
    pretrained = 'weights/vgg16_reducedfc.pth',
    m2det_config = dict(
        backbone = 'vgg16',
        net_family = 'vgg', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        base_out = [22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = False,
        smooth = True,
        num_classes = 81,
        ),
    rgb_means = (104, 117, 123),
    p = 0.6,
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 117, 176], # if you want to set this with custom manner, compute by input size / each map's size.
        #size_pattern = [0.035, 0.085, 0.176, 0.352, 0.713, 0.99, 1.19],
        size_pattern = [0.04, 0.11, 0.23, 0.39, 0.58, 0.80, 1.05],
        ),
    save_eposhs = 10,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 5,
    per_batch_size = 11,
    lr = [0.004, 0.002, 0.0004, 0.00004, 0.000004],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        COCO = [90, 110, 130, 150, 160],
        VOC = [100, 150, 200, 250, 300],
        ),
    print_epochs = 10,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
    keep_per_class = 50,
    save_folder = 'eval'
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC = dict(
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),
    COCO = dict(
        train_sets = [('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets = [('2014', 'minival')],
        test_sets = [('2015', 'test-dev')],
        )
    )

import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home,"data/VOCdevkit/")
COCOroot = os.path.join(home,"data/coco/")
