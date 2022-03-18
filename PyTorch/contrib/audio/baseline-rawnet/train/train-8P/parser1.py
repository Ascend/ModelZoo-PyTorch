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

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import argparse
import os.path


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args():
    parser = argparse.ArgumentParser()

    #dir
    #默认文件位置
    # parser.add_argument('-name', type = str, required = True)
    parser.add_argument('-name', type = str, default = 'NPU-8P-4')
    parser.add_argument('-save_dir', type = str, default = 'DNNs/')
    parser.add_argument('-save_profile', type = str, default = 'prof/') #保存/prof文件
    parser.add_argument('-save_log', type = str, default = 'log/') #保存训练日志
    parser.add_argument('-DB', type = str, default = 'DB/VoxCeleb1/')
    parser.add_argument('-DB_vox2', type = str, default = 'DB/VoxCeleb2/')
    parser.add_argument('-dev_wav', type = str, default = 'wav/')
    parser.add_argument('-val_wav', type = str, default = 'dev_wav/')
    parser.add_argument('-eval_wav', type = str, default = 'eval_wav/')
    
    #hyper-params
    parser.add_argument('-frame', type = int, default = 135395880960) # the number of frames of all the .wav file
    parser.add_argument('-bs', type = int, default = 1024)
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-nb_samp', type = int, default = 59049)
    parser.add_argument('-window_size', type = int, default = 11810)
    
    parser.add_argument('-wd', type = float, default = 0.0001) #初始化权重
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-optimizer', type = str, default = 'Adam')
    parser.add_argument('-nb_worker', type = int, default = 8)
    parser.add_argument('-temp', type = float, default = .5)
    parser.add_argument('-seed', type = int, default = 1234) 
    parser.add_argument('-nb_val_trial', type = int, default = 40000) 
    parser.add_argument('-lr_decay', type = str, default = 'keras')
    parser.add_argument('-load_model_dir', type = str, default = '')
    parser.add_argument('-load_model_opt_dir', type = str, default = '')


    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 59049)
    
    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-make_val_trial', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-debug', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-comet_disable', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-mg', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-load_model', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-use_prof', type = str2bool, nargs='?', const=False, default = True) #默认情况下训练不使用prof文件，因此设置默认值为false
    parser.add_argument('-amp_mode', type = str2bool, nargs='?', const=True, default = True) #使用Apex进行加速，为了进行混合精度训练，我们默认其为True

    # NPU修改
    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--addr', default='127.0.0.1', type=str, help='master addr')
    parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
    parser.add_argument('--amp', default=False, action='store_true', help='use amp to train the model')
    parser.add_argument('--loss_scale', default=1024., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O2', type=str,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--world_size', default='8', type=int)
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    return args