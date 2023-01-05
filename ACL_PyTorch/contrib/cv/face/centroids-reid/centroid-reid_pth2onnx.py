# Copyright 2022 Huawei Technologies Co., Ltd
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

import sys
sys.path.append("./centroids-reid")
import argparse
from config import cfg
from train_ctl_model import CTLModel
import torch
import torch.onnx

def main(args):    
    model = CTLModel  
    model = model.load_from_checkpoint(
        args.input_file
    )
    input_sample=torch.randn((args.batch_size,3,256,128)) 
    model.eval()
    model.forward=model.test_step
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}} 
    torch.onnx.export(model, input_sample, args.output_file, export_params=True,\
    input_names = ['input'], output_names = ['output'], dynamic_axes = dynamic_axes,opset_version=11, verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model to ONNX")
    parser.add_argument("--output_file", default="centroid-reid_r50_bs1.onnx", help="name of onnx", type=str)
    parser.add_argument("--input_file", default="./models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt", type=str)
    parser.add_argument("--batch_size", default=1, help="batch_size", type=int)
    args = parser.parse_args()
    main(args)