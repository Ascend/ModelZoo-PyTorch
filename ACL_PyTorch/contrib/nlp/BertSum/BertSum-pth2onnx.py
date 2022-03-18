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
import torch
import os
import sys
import argparse
import numpy as np
from pytorch_pretrained_bert import BertConfig
from models.model_builder import Summarizer

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    input_names=['src','segs','clss','mask','mask_cls']
    output_names = ["output"]
    onnx_path = args.onnx_path
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    checkpoint = torch.load(args.path, map_location='cpu')
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()
    cur_path = os.getcwd()    
    src = np.fromfile(f'{cur_path}/pre_data/src/data_1.bin', dtype=np.int64)
    segs = np.fromfile(f'{cur_path}/pre_data/segs/data_1.bin', dtype=np.int64)
    clss = np.fromfile(f'{cur_path}/pre_data/clss/data_1.bin', dtype=np.int64)
    mask = np.fromfile(f'{cur_path}/pre_data/mask/data_1.bin', dtype=np.bool_)
    mask_cls = np.fromfile(f'{cur_path}/pre_data/mask_cls/data_1.bin', dtype=np.bool_)
    print(src.shape)
    print(segs.shape)
    print(clss.shape)
    print(mask.shape)
    print(mask_cls.shape)
    #-----------------------------13000-----------------------------
    dummy_input0 = torch.from_numpy(src).reshape(1,512)
    dummy_input1 = torch.from_numpy(segs).reshape(1,512)
    dummy_input2 = torch.from_numpy(clss).reshape(1,37)
    dummy_input3 = torch.from_numpy(mask).reshape(1,512)
    dummy_input4 = torch.from_numpy(mask_cls).reshape(1,37)
    #--------------------------------------------------------------------'''
    torch.onnx.export(model,(dummy_input0,dummy_input1,dummy_input2,dummy_input3,dummy_input4),onnx_path,input_names = input_names,output_names=output_names,verbose=True,opset_version=9)
    
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
    main(args)