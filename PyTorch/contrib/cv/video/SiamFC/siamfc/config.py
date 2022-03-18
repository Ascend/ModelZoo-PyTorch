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

class Config:
    # dataset related
    exemplar_size = 127                    # exemplar size   z
    instance_size = 255                    # instance size   x
    context_amount = 0.5                   # context amount

    # training related
    num_per_epoch = 51200                  # num of samples per epoch, original 53200
    train_ratio = 0.9                      # training ratio of VID dataset
    frame_range = 100                      # frame range of choosing the instance
    train_batch_size = 32                  # training batch size
    valid_batch_size = 32                  # validation batch size
    train_num_workers = 4                  # number of workers of train dataloader
    valid_num_workers = 4                  # number of workers of validation dataloader
    lr = 0.06                              # learning rate of SGD
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0                       # weight decay of optimizator
    step_size = 18                         # step size of LR_Schedular
    gamma = 0.1                            # decay rate of LR_Schedular
    epoch = 50                             # total epoch
    seed = 1234                            # seed to sample training videos
    log_dir = './models/logs'              # log dirs
    radius = 16                            # radius of positive label
    response_scale = 1e-3                  # normalize of response
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 3                          # number of scales
    scale_lr = 0.59                        # scale learning rate
    response_up_stride = 16                # response upsample stride
    response_sz = 17                       # response size
    train_response_sz = 15                 # train response size
    window_influence = 0.176               # window influence
    scale_penalty = 0.9745                 # scale penalty
    total_stride = 8                       # total stride of backbone
    sample_type = 'uniform'
    gray_ratio = 0.25
    blur_ratio = 0.15

    # test related
    model_path = './models/siamfc_50.pth'
    gpu_id = 1


config = Config()
