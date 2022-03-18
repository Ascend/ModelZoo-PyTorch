# Copyright 2021 Huawei Technologies Co., Ltd
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

# Copyright 2020 Huawei Technologies Co., Ltd
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

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", default="res101", type=str)
    parser.add_argument("--dataset", default="UCAS_AOD", type=str)
    parser.add_argument("--train_path", default="UCAS_AOD/train.txt", type=str)
    parser.add_argument("--test_path", default="UCAS_AOD/test.txt", type=str)
    parser.add_argument("--root_path", default="datasets/evaluate", type=str)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--training_size", default="800,1344", type=str)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--pos_max_num", default=1024, type=int)
    parser.add_argument("--augment", default=0, type=int)

    parser.add_argument("--distributed", default=1, type=int)
    parser.add_argument("--MASTER_ADDR", default="127.0.0.1", type=str)
    parser.add_argument("--MASTER_PORT", default="29688", type=str)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--device_list", default="0,1,2,3,4,5,6,7", type=str)
    parser.add_argument("--dist_num", default=1, type=int)
    parser.add_argument("--dist_index", default=0, type=int)
    parser.add_argument("--npus_per_node", default=8, type=int)
    parser.add_argument("--n_threads", default=64, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr0", default=0.0001, type=float)
    parser.add_argument("--warmup_lr", default=0.00001, type=float)
    parser.add_argument("--warm_epoch", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--begin_epoch", default=0, type=int)

    parser.add_argument("--work_dir", default="weights", type=str)
    parser.add_argument("--file_hash", default="result", type=str)
    parser.add_argument("--start_save", default=50, type=int)
    parser.add_argument("--save_interval", default=-1, type=int)
    parser.add_argument("--start_test", default=50, type=int)
    parser.add_argument("--test_interval", default=-1, type=int)
    parser.add_argument("--inference", default=0, type=int)
    parser.add_argument("--amp_cfg", default=1, type=int)
    parser.add_argument("--opt_level", default="O1", type=str)
    parser.add_argument("--manual_seed", default=0, type=int)
    args = parser.parse_args()
    return args


def parse_opts_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", default="res101", type=str)
    parser.add_argument("--dataset", default="UCAS_AOD", type=str)
    parser.add_argument("--target_size", default="800,1344", type=str)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--weight", default="./weights/best.pth", type=str)
    parser.add_argument("--test_path", default="UCAS_AOD/test.txt", type=str)
    parser.add_argument("--root_path", default="datasets/evaluate", type=str)
    parser.add_argument("--device_index", default=0, type=int)
    args = parser.parse_args()
    return args
