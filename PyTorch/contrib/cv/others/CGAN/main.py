# Copyright 2020 Huawei Technologies Co., Ltd
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
import argparse, os, torch
from CGAN import CGAN
import torch.distributed as dist

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN', 'LSGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--loss_scale', type=float, default=128.0)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--amp', type=int, default=0)
    parser.add_argument('--opt_level', type=str, default="02")
    parser.add_argument("--is_distributed", type=int, default=0, help='choose ddp or not')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--DeviceID', type=str, default="0")
    parser.add_argument('--world_size', type=int, default=8)
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    
    args = parse_args()
    if args is None:
        exit()

    if args.is_distributed == 0:
        local_device = torch.device(f'npu:{args.DeviceID}')
        torch.npu.set_device(local_device)
        print("using npu :{}".format(args.DeviceID))
    else:
        # os.environ['MASTER_ADDR'] = '127.0.0.2'
        # os.environ['MASTER_PORT'] = '29689'
        # os.environ['WORLD_SIZE'] = '4'
        dist.init_process_group(backend='hccl',world_size=args.world_size, rank=args.local_rank)
        local_device = torch.device(f'npu:{args.local_rank}')
        torch.npu.set_device(local_device)
        if args.local_rank == 0:
            print("using npu :{}".format(args.DeviceID))
        # declare instance for GAN
    if args.gan_type == 'CGAN':
        gan = CGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    if args.local_rank == 0:
        print(" [*] Training finished!")
        gan.visualize_results(args.epoch)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
