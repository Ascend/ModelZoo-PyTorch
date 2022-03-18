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
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import setproctitle
from datetime import datetime
from pathlib import Path
from imnet_resnet50_scratch import TrainerConfig, ClusterConfig, Trainer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def run(input_sizes, learning_rate, epochs, batch, node, workers, imnet_path, shared_folder_path, job_id, local_rank, global_rank, num_tasks):
    cluster_cfg = ClusterConfig(dist_backend="hccl", dist_url="env://")
    shared_folder = None
    data_folder_Path = None
    if Path(str(shared_folder_path)).is_dir():
        shared_folder = Path(shared_folder_path + "/training/")
    else:
        raise RuntimeError("No shared folder available")
    if Path(str(imnet_path)).is_dir():
        data_folder_Path = Path(str(imnet_path))
    else:
        raise RuntimeError("No shared folder available")
    train_cfg = TrainerConfig(
        data_folder=str(data_folder_Path),
        epochs=epochs,
        lr=learning_rate,
        input_size=input_sizes,
        batch_per_gpu=batch,
        save_folder=str(shared_folder_path),
        workers=workers,
        imnet_path=imnet_path,
        local_rank=local_rank,
        global_rank=global_rank,
        num_tasks=num_tasks,
        job_id=job_id,
    )

    # Create the executor
    os.makedirs(str(shared_folder), exist_ok=True)
    init_file = shared_folder / datetime.now().strftime("%Y%m%d-%H%M%S")
    if init_file.exists():
        os.remove(str(init_file))

    trainer = Trainer(train_cfg, cluster_cfg)

    # The code should be launch on each GPUs
    try:
        if local_rank == 0:
            val_accuracy = trainer.__call__()
            print(f"Validation accuracy: {val_accuracy}")
        else:
            trainer.__call__()
    except:
        print("Job failed")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for ResNet50 FixRes",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', default=0.025,
                        type=float, help='base learning rate')
    parser.add_argument('--input_size', default=224,
                        type=int, help='images input size')
    parser.add_argument('--epochs', default=120, type=int, help='epochs')
    parser.add_argument('--batch', default=64, type=int, help='Batch by GPU')
    parser.add_argument('--node', default=1, type=int, help='GPU nodes')
    parser.add_argument('--workers', default=10,
                        type=int, help='Numbers of CPUs')
    parser.add_argument('--imnet_path', default='/opt/npu/imagenet/',
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--shared_folder_path',
                        default='./train_cache/', type=str, help='Shared Folder')
    parser.add_argument('--local_rank', default=0,
                        type=int, help='GPU: Local rank')
    parser.add_argument('--global_rank', default=0,
                        type=int, help='GPU: glocal rank')
    parser.add_argument('--num_tasks', default=8, type=int,
                        help='How many GPUs are used')
    parser.add_argument("--addr", default="127.0.0.1", type=str)
    args = parser.parse_args()
    setproctitle.setproctitle('FIXRES - train')
    args.job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'
    run(args.input_size, args.learning_rate, args.epochs, args.batch, args.node, args.workers,
        args.imnet_path, args.shared_folder_path, args.job_id, args.local_rank, args.global_rank, args.num_tasks)
