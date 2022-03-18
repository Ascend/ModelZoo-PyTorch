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
import argparse
import os
from solver_1p import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import sys
import torch
import moxing as mox
from collections import OrderedDict
from network import R2AttU_Net

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    real_path = sys.path[0]
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(config.data_url, real_path)

    config.train_path = os.path.join(real_path, "train")
    config.valid_path = os.path.join(real_path, "valid")
    config.test_path = os.path.join(real_path, "test")
    

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=1,   # test mode, batchsize default is 1.
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    config.num_epochs = 1
    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    unet_path = os.path.join(config.result_path, 'final.pkl')
    final_unet = solver.unet.state_dict()
    torch.save(final_unet, unet_path)
    onnx_file_path = os.path.join(config.result_path, 'R2AttU_Net.onnx')

    model = R2AttU_Net(img_ch=3,output_ch=1,t=2)
    model.load_state_dict(torch.load(unet_path, map_location="cpu"), strict=False)
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(4, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names,
                      opset_version=11)
    print("export R2AttU_Net.onnx success")
    mox.file.copy_parallel(config.result_path, config.train_url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--test_model_path', type=str, default='./models')
    parser.add_argument('--data_url', type=str, default='./dataset/train/')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--pretrain',  type=int, default=0)
    parser.add_argument('--pretrain_path',  type=str, default="")
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--use_apex', type=int, default=1)
    parser.add_argument('--apex_level', type=str, default="O2")
    parser.add_argument('--loss_scale', type=float, default=128.)
    parser.add_argument('--train_url', type=str, default=None)



    config = parser.parse_args()
    main(config)
