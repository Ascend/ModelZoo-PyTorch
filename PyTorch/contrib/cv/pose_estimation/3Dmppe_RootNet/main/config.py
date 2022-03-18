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
import sys
import numpy as np


class Config:
    
    ## dataset
    # training set
    # 3D: Human36M, MuCo, PW3D
    # 2D: MSCOCO, MPII 
    # Note that list must consists of one 3D dataset (first element of the list) + several 2D datasets
    trainset_3d = ['MuCo']
    trainset_2d = ['MPII']

    # testing set
    # Human36M, MuPoTS, MSCOCO, PW3D
    testset = 'MuPoTS'

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    prof_dir = osp.join(output_dir, 'prof')
 
    ## model setting
    resnet_type = 50 # 50, 101, 152
    
    ## input, output
    input_shape = (256, 256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    bbox_real = (2000, 2000) # Human36M, MuCo, MuPoTS: (2000, 2000), PW3D: (2, 2)

    ## training config
    lr_dec_epoch = [17] # The learning rate will drop from this epoch
    end_epoch = 20
    lr = 0.001
    lr_dec_factor = 10
    batch_size = 32
    continue_train = False
    use_prof = False
    distributed = False
    world_size = 1
    num_thread = 16
    npus_per_node = 1
    loss_scale = None
    opt_level = None
    npu_device = '0'
    data_path = 'data'
    
    #8P
    addr = '127.0.0.1'
    device_list = '0,1,2,3,4,5,6,7'
    amp = False
    port = '50000'
    
    ## testing config
    test_batch_size = 32
    use_gt_bbox = True


    def set_args_train8P(self, resnet_type, lr_dec_epoch, end_epoch, lr, lr_dec_factor, batch_size,
                         continue_train, distributed, world_size, num_thread, use_prof,npus_per_node,
                         addr, device_list, amp, loss_scale, opt_level, prot, data_path):
        
        ## model setting
        self.resnet_type = resnet_type
        
        ## training config
        self.lr_dec_epoch = [lr_dec_epoch]
        self.end_epoch = end_epoch
        self.lr = lr
        self.lr_dec_factor = lr_dec_factor
        self.batch_size = batch_size
        self.continue_train = continue_train
        self.use_prof = use_prof
        self.distributed = distributed
        self.world_size = world_size
        self.num_thread = num_thread
        self.npus_per_node = npus_per_node
        self.loss_scale = loss_scale
        self.opt_level = opt_level
        self.data_path = data_path
        
        ## 8P
        self.addr = addr
        self.prot = prot
        self.device_list = device_list
        self.amp = amp
        
        
    def set_args_train1P(self, resnet_type, lr_dec_epoch, end_epoch, lr, lr_dec_factor, batch_size, 
                       rank, continue_train, distributed, world_size, num_thread, use_prof, 
                       npus_per_node, amp, loss_scale, opt_level, npu_device, data_path):
        
        ## model setting
        self.resnet_type = resnet_type
        
        ## training config
        self.lr_dec_epoch = [lr_dec_epoch]
        self.end_epoch = end_epoch
        self.lr = lr
        self.lr_dec_factor = lr_dec_factor
        self.batch_size = batch_size
        self.rank = rank
        self.continue_train = continue_train
        self.use_prof = use_prof
        self.distributed = distributed
        self.world_size = world_size
        self.num_thread = num_thread
        self.npus_per_node = npus_per_node
        self.loss_scale = loss_scale
        self.opt_level = opt_level
        self.amp = amp
        self.npu_device = npu_device
        self.data_path = data_path


    def set_args_test(self, data_path):
        self.data_path = data_path


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))

from utils.dir_utils import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
make_folder(cfg.prof_dir)
