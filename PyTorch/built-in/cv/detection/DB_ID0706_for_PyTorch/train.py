#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

#!python3

import argparse
import time
import os
import ast
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import yaml
import torch.distributed as dist

from trainer import Trainer
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config

def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id
    return process_device_map

def main(args):
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    os.environ["MASTER_ADDR"] = args["addr"]
    os.environ["MASTER_PORT"] = args["Port"]
    if args["dist_url"] == "env://" and args["world_size"] == -1:
        args["world_size"] = int(os.environ["WORLD_SIZE"])

    args["process_device_map"] = device_id_to_process_device_map(args["device_list"])
    if torch.npu.is_available():
        npus_per_node = len(args["process_device_map"])
    else:
        npus_per_node = torch.cuda.device_count()
    print("{} node found.".format(npus_per_node))
    if args['distributed']:
        args["world_size"] = npus_per_node * args["world_size"]
        torch.multiprocessing.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, args))
    else:
        main_worker(0, npus_per_node, args)

def main_worker(dev, npus_per_node, args):
    os.environ['PROFILE_TYPE'] = args.prof_type
    if args.prof_type == 'GE':
        os.environ['GE_PROFILING_TO_STD_OUT'] = '1'
    if args['bin']:
        torch.npu.set_compile_mode(jit_compile=False)
        print("use bin train model")
    device_id = args["process_device_map"][dev]
    args["device_id"] = device_id
    loc = "npu:{}".format(device_id)
    if torch.npu.is_available():
        torch.npu.set_device(loc)
    if args["distributed"]:
        if args["dist_url"] == "env://" and args.local_rank == -1:
            args["local_rank"] = int(os.environ["RANK"])
        args["local_rank"] = args["local_rank"] * npus_per_node + dev
        print("local_rank:", args["local_rank"])
        dist.init_process_group(backend=args["dist_backend"],
                                world_size=args["world_size"],
                                rank=args["local_rank"])
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment_args["train"]["data_loader"]["dataset"]["data_dir"] = [args["data_path"]]
    experiment_args["train"]["data_loader"]["dataset"]["data_list"] = [os.path.join(
        args["data_path"], "train_list.txt")]
    experiment_args["validation"]["data_loaders"]["icdar2015"]["dataset"]["data_dir"] = [args["data_path"]]
    experiment_args["validation"]["data_loaders"]["icdar2015"]["dataset"]["data_list"] = [os.path.join(
        args["data_path"], "test_list.txt")]
    experiment_args["evaluation"]["data_loaders"]["icdar2015"]["dataset"]["data_dir"] = [args["data_path"]]
    experiment_args["evaluation"]["data_loaders"]["icdar2015"]["dataset"]["data_list"] = [os.path.join(
        args["data_path"], "test_list.txt")]
    experiment = Configurable.construct_class_from_config(experiment_args)

    if not args['print_config_only']:
        trainer = Trainer(experiment)
        print("start train in device: ", device_id)
        trainer.train(args["profiling"], args["start_step"], args["stop_step"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--start_iter', type=int, help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--start_epoch', type=int, help='Begin counting epoch starting from this value (should be used with resume)')
    parser.add_argument('--max_size', type=int, help='max length of label')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, help='The optimizer want to use')
    parser.add_argument('--thresh', type=float, help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    parser.add_argument('--visualize', action='store_true', help='visualize maps in tensorboard')
    parser.add_argument('--force_reload', action='store_true', dest='force_reload', help='Force reload data meta')
    parser.add_argument('--no-force_reload', action='store_false', dest='force_reload', help='Force reload data meta')
    parser.add_argument('--validate', action='store_true', dest='validate', help='Validate during training')
    parser.add_argument('--no-validate', action='store_false', dest='validate', help='Validate during training')
    parser.add_argument('--print-config-only', action='store_true', help='print config without actual training')
    parser.add_argument('--debug', action='store_true', dest='debug', help='Run with debug mode, which hacks dataset num_samples to toy number')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Run without debug mode')
    parser.add_argument('--benchmark', action='store_true', dest='benchmark', help='Open cudnn benchmark mode')
    parser.add_argument('--no-benchmark', action='store_false', dest='benchmark', help='Turn cudnn benchmark mode off')
    parser.add_argument('-d', '--distributed', action='store_true', dest='distributed', help='Use distributed training')
    parser.add_argument('--amp', default=False, action='store_true', help='Use amp in the model training')
    parser.add_argument('--dist_url', default="tcp://224.66.41.62:23456", type=str, help='url used to set up distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=4, type=int, help='The number of accessible gpus')
    parser.add_argument('--device_list', dest="device_list", default="0", type=str, help='Use distributed training')
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist_backend', default="hccl", type=str, help='distributed backend')
    parser.add_argument('--addr', default="90.90.176.102", type=str, help='master addr')
    parser.add_argument('--Port', default="29500", type=str, help='master Port')
    parser.add_argument('--bin',  type=ast.literal_eval, default=False ,help='turn on bin')
    parser.add_argument('--profiling', default='', type=str, help='type of profiling')
    parser.add_argument('--start_step', default=-1, type=int, help='number of start step')
    parser.add_argument('--stop_step', default=-1, type=int, help='number of stop step')
    parser.add_argument("--prof_type", default='None',
                       	 choices=['TORCH', 'CANN', 'GE', 'None'],
                       	 help="The type of profile.")
    parser.set_defaults(debug=False)
    parser.set_defaults(benchmark=True)

    args = parser.parse_args()

    main(args)
