# Copyright 2022 Huawei Technologies Co., Ltd
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
import moxing as mox
from solver import Solver
from data_loader import get_loader
from data_loader import get_dist_loader
import numpy as np
from torch.backends import cudnn
import random
import sys
import torch
from collections import OrderedDict
from network import AttU_Net


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
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    config.is_master_node = not config.distributed or config.npu_idx == 0
    # cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

        # Create directories if not exist
    if config.is_master_node:
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path)
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
        config.result_path = os.path.join(config.result_path, config.model_type)
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)

    if config.distributed:
        if 'RANK_SIZE' in os.environ and 'RANK_ID' in os.environ:
            config.rank_size = int(os.environ['RANK_SIZE'])
            config.rank_id = int(os.environ['RANK_ID'])
            config.rank = config.dist_rank * config.rank_size + config.rank_id
            config.world_size = config.world_size * config.rank_size
            config.npu_idx = config.rank_id
            config.batch_size = int(config.batch_size / config.rank_size)
            config.num_workers = int((config.num_workers + config.rank_size - 1) / config.rank_size)
        else:
            raise RuntimeError("init_distributed_mode failed.")

        torch.distributed.init_process_group(backend='hccl', init_method=config.dist_url,
                                             world_size=config.world_size, rank=config.rank)
    else:
        config.rank_size = 1

    real_path = sys.path[0]
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    mox.file.copy_parallel(config.data_url, real_path)

    config.train_path = os.path.join(real_path, "train")
    config.valid_path = os.path.join(real_path, "valid")
    config.test_path = os.path.join(real_path, "test")

    if config.distributed:
        train_loader, train_sampler = get_dist_loader(image_path=config.train_path,
                                                      image_size=config.image_size,
                                                      batch_size=config.batch_size,
                                                      num_workers=config.num_workers,
                                                      mode='train',
                                                      augmentation_prob=config.augmentation_prob)

        config.train_sampler = train_sampler
    else:

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
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    unet_path = os.path.join(config.result_path, 'final.pkl')
    final_unet = solver.unet.state_dict()
    torch.save(final_unet, unet_path)
    onnx_file_path = os.path.join(config.result_path, 'AttU_Net.onnx')

    model = AttU_Net(img_ch=3, output_ch=1)
    model.load_state_dict(torch.load(unet_path, map_location="cpu"), strict=False)
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(4, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_file_path, input_names=input_names, output_names=output_names,
                      opset_version=11)
    print("export AttU_Net.onnx success")
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
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='AttU_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='/home/dataset/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--data_url', type=str, default='/home/dataset/')
    parser.add_argument('--train_url', type=str, default=None)

    parser.add_argument('--npu_idx', type=int, default=0)
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex_opt_level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument('--loss_scale_value',
                        default=1024,
                        type=int,
                        help='set loss scale value.')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pth_path', type=str, default='/home/AttU_Net.pkl')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed',
                        default=12345,
                        type=int,
                        help='Manually set random seed')
    config = parser.parse_args()
    main(config)
