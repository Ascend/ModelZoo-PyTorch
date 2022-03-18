# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import torch

from models import data_loader
from models.data_loader import load_dataset
from models.trainer import build_trainer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_max_shape(test_iter):
    max_shape_1=0
    max_shape_2=0
    for batch in test_iter:
        if batch.src.shape[1] > max_shape_1:
            max_shape_1 = batch.src.shape[1]
        if batch.clss.shape[1] > max_shape_2:
            max_shape_2 = batch.clss.shape[1]
        #print(batch.src[0].shape)
    return max_shape_1,max_shape_2
    
def preprocess(args,device):
    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    test_iter1 =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    cur_path = os.getcwd()
    main_path = cur_path + '/pre_data'
    main_path_1 = cur_path + '/pre_data_1'
    i=0
    if not os.path.exists(os.path.join(cur_path,'pre_data')):  ###########first inference
        os.makedirs(os.path.join(cur_path,'pre_data'))    
    if not os.path.exists(os.path.join(main_path,'src')):
        os.makedirs(os.path.join(main_path,'src'))
    if not os.path.exists(os.path.join(main_path,'segs')):
        os.makedirs(os.path.join(main_path,'segs'))
    if not os.path.exists(os.path.join(main_path,'clss')):
        os.makedirs(os.path.join(main_path,'clss'))
    if not os.path.exists(os.path.join(main_path,'mask')):
        os.makedirs(os.path.join(main_path,'mask'))
    if not os.path.exists(os.path.join(main_path,'mask_cls')):
        os.makedirs(os.path.join(main_path,'mask_cls'))
    
    if not os.path.exists(os.path.join(cur_path,'pre_data_1')):        ###########second inference
        os.makedirs(os.path.join(cur_path,'pre_data_1'))
    if not os.path.exists(os.path.join(main_path_1,'src')):
        os.makedirs(os.path.join(main_path_1,'src'))
    if not os.path.exists(os.path.join(main_path_1,'segs')):
        os.makedirs(os.path.join(main_path_1,'segs'))
    if not os.path.exists(os.path.join(main_path_1,'clss')):
        os.makedirs(os.path.join(main_path_1,'clss'))
    if not os.path.exists(os.path.join(main_path_1,'mask')):
        os.makedirs(os.path.join(main_path_1,'mask'))
    if not os.path.exists(os.path.join(main_path_1,'mask_cls')):
        os.makedirs(os.path.join(main_path_1,'mask_cls'))
    max_shape_1,max_shape_2 = get_max_shape(test_iter)
    print(max_shape_1,max_shape_2)
    #############################above get max dimension ###########################
    for batch in test_iter1:
        if batch.src.shape[0]==2:
            if batch.src[0].shape[0] < max_shape_1:
                add_zero = (torch.zeros([batch.src.shape[0],max_shape_1-batch.src[0].shape[0]])).long()  #######change to int64
                add_bool = torch.zeros([batch.src.shape[0],max_shape_1-batch.src[0].shape[0]],dtype=torch.bool)
                batch.src = torch.cat([batch.src,add_zero],dim=1)
                batch.segs = torch.cat([batch.segs,add_zero],dim=1)
                batch.mask = torch.cat([batch.mask,add_bool],dim=1)
            if batch.clss[0].shape[0] < max_shape_2:
                add_zero = (torch.zeros([batch.clss.shape[0],max_shape_2-batch.clss[0].shape[0]])).long()       #######change to int64
                add_bool = torch.zeros([batch.clss.shape[0],max_shape_2-batch.clss[0].shape[0]],dtype=torch.bool)
                batch.clss = torch.cat([batch.clss,add_zero],dim=1)
                batch.mask_cls = torch.cat([batch.mask_cls,add_bool],dim=1)
            ##############first dimension
            batch.src[0].numpy().tofile(os.path.join(main_path,'src','data_'+str(i)+'.bin'))
            batch.segs[0].numpy().tofile(os.path.join(main_path,'segs','data_'+str(i)+'.bin'))
            batch.clss[0].numpy().tofile(os.path.join(main_path,'clss','data_'+str(i)+'.bin'))
            batch.mask[0].numpy().tofile(os.path.join(main_path,'mask','data_'+str(i)+'.bin'))
            batch.mask_cls[0].numpy().tofile(os.path.join(main_path,'mask_cls','data_'+str(i)+'.bin'))
            #############second dimension
            batch.src[1].numpy().tofile(os.path.join(main_path_1,'src','data_'+str(i)+'.bin'))
            batch.segs[1].numpy().tofile(os.path.join(main_path_1,'segs','data_'+str(i)+'.bin'))
            batch.clss[1].numpy().tofile(os.path.join(main_path_1,'clss','data_'+str(i)+'.bin'))
            batch.mask[1].numpy().tofile(os.path.join(main_path_1,'mask','data_'+str(i)+'.bin'))
            batch.mask_cls[1].numpy().tofile(os.path.join(main_path_1,'mask_cls','data_'+str(i)+'.bin'))
        else:       
            #print(batch.clss.dtype)
            if batch.src[0].shape[0] < max_shape_1:
                add_zero = (torch.zeros([batch.src.shape[0],max_shape_1-batch.src[0].shape[0]])).long()      #######change to int64
                add_bool = torch.zeros([batch.src.shape[0],max_shape_1-batch.src[0].shape[0]],dtype=torch.bool)
                batch.src = torch.cat([batch.src,add_zero],dim=1)
                batch.segs = torch.cat([batch.segs,add_zero],dim=1)
                batch.mask = torch.cat([batch.mask,add_bool],dim=1)
            if batch.clss[0].shape[0] < max_shape_2:
                add_zero = (torch.zeros([batch.clss.shape[0],max_shape_2-batch.clss[0].shape[0]])).long()       #######change to int64
                add_bool = torch.zeros([batch.clss.shape[0],max_shape_2-batch.clss[0].shape[0]],dtype=torch.bool)
                batch.clss = torch.cat([batch.clss,add_zero],dim=1)
                batch.mask_cls = torch.cat([batch.mask_cls,add_bool],dim=1)
            batch.src.numpy().tofile(os.path.join(main_path,'src','data_'+str(i)+'.bin'))
            batch.segs.numpy().tofile(os.path.join(main_path,'segs','data_'+str(i)+'.bin'))
            batch.clss.numpy().tofile(os.path.join(main_path,'clss','data_'+str(i)+'.bin'))
            batch.mask.numpy().tofile(os.path.join(main_path,'mask','data_'+str(i)+'.bin'))
            batch.mask_cls.numpy().tofile(os.path.join(main_path,'mask_cls','data_'+str(i)+'.bin'))
        i = i+1

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='classifier', type=str, choices=['classifier','transformer','rnn','baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train','validate','test'])
    parser.add_argument("-bert_data_path", default='../bert_data')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-bert_config_path", default='../bert_config_uncased_base.json')

    parser.add_argument("-batch_size", default=1000, type=int)

    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-onnx_path", default="")
    parser.add_argument("-path", default="")
    
    args = parser.parse_args()
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = -1 if device == "cpu" else 0

    preprocess(args,device)