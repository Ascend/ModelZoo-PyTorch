from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import torch.utils.data
from opts_pose import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from datasets.sample.multi_pose import Multiposebatch

def main(onnx_path,path):
    opt = opts().parse()
    input_names=["image"]
    output_names = ["output1","output2","output3","output4"]
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model , path, None, opt.resume, opt.lr, opt.lr_step)
    dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'},'output2': {0: '-1'},'output3': {0: '-1'},'output4': {0: '-1'}}
    model.eval()
    dummy_input = torch.randn(1,3,800,800)
    torch.onnx.export(model,dummy_input,onnx_path,export_params=True,dynamic_axes = dynamic_axes,input_names = input_names,output_names = output_names,verbose=True) 

if __name__ =="__main__":
    #onnx_path = sys.argv[1]
    #path = sys.argv[2]
    onnx_path = '../CenterFace.onnx'
    path = '../model_best.pth'
    main(onnx_path,path)
