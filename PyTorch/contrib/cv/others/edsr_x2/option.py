# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import argparse

parser = argparse.ArgumentParser(description="EDSR and MDSR")

parser.add_argument("--arch", type=str, default="EDSR_x2")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--device", type=str, default="npu", help="npu or gpu")
parser.add_argument("--device_list", type=str, default="0,1,2,3,4,5,6,7")
parser.add_argument(
    "--world_size", type=int, default=1, help="number of nodes for distributed training"
)
parser.add_argument(
    "--ifcontinue",
    action="store_true",
    default=False,
    help="if continue to train the model",
)
parser.add_argument(
    "--checkpoint_path", type=str, help="the path of checkpoint to load"
)
parser.add_argument("--dist-backend", type=str)

# amp setting
parser.add_argument("--amp", action="store_true",
                    help="Use amp to train the model")
parser.add_argument("--loss_scale",    default="dynamic",
                    type=str,    help="amp setting: loss scale, default 128.0")
parser.add_argument("--opt_level", default="O2", type=str,
                    help="amp setting: opt level, default O2")

# Hardware specifications
parser.add_argument("--cpu", action="store_true", help="use cpu only")
parser.add_argument("--n_GPUs", type=int, default=1, help="number of GPUs")
parser.add_argument("--seed", type=int, default=1, help="random seed")
# npu setting
parser.add_argument("--use_npu", action="store_true",
                    help="Use NPU to train the model")
parser.add_argument("--npu", default=0, type=int, help="NPU id to use")


# Data specifications
parser.add_argument("--dir_data", type=str,
                    default="../../../dataset", help="dataset directory")
parser.add_argument("--dir_demo", type=str,
                    default="test", help="demo image directory")
parser.add_argument("--data_train", type=str,
                    default="DIV2K", help="train dataset name")
parser.add_argument("--data_test", type=str,
                    default="DIV2K", help="test dataset name")
parser.add_argument("--data_range", type=str,
                    default="1-800/801-810", help="train/test data range")
parser.add_argument("--ext", type=str, default="sep",
                    help="dataset file extension")
parser.add_argument("--scale", type=str, default="2",
                    help="super resolution scale")
parser.add_argument("--patch_size", type=int,
                    default=192, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=255,
                    help="maximum value of RGB")
parser.add_argument(
    "--n_colors", type=int, default=3, help="number of color channels to use"
)
parser.add_argument(
    "--no_augment", action="store_true", help="do not use data augmentation"
)

# Model specifications
parser.add_argument("--model", default="EDSR", help="model name")

parser.add_argument("--act", type=str, default="relu",
                    help="activation function")
parser.add_argument(
    "--pre_train", type=str, default="", help="pre-trained model directory"
)
parser.add_argument(
    "--extend", type=str, default=".", help="pre-trained model directory"
)
parser.add_argument(
    "--n_resblocks", type=int, default=32, help="number of residual blocks"
)
parser.add_argument("--n_feats", type=int, default=256,
                    help="number of feature maps")
parser.add_argument("--res_scale", type=float,
                    default=0.1, help="residual scaling")
parser.add_argument(
    "--shift_mean", default=True, help="subtract pixel mean from the input"
)
parser.add_argument("--dilation", action="store_true",
                    help="use dilated convolution")

# Option for Residual dense network (RDN)
parser.add_argument(
    "--G0", type=int, default=64, help="default number of filters. (Use in RDN)"
)
parser.add_argument(
    "--RDNkSize", type=int, default=3, help="default kernel size. (Use in RDN)"
)
parser.add_argument(
    "--RDNconfig", type=str, default="B", help="parameters config of RDN. (Use in RDN)"
)

# Option for Residual channel attention network (RCAN)
parser.add_argument(
    "--n_resgroups", type=int, default=10, help="number of residual groups"
)
parser.add_argument(
    "--reduction", type=int, default=16, help="number of feature maps reduction"
)

# Training specifications
parser.add_argument(
    "--test_every", type=int, default=1000, help="do test per every N batches"
)
parser.add_argument("--epochs", type=int, default=300,
                    help="number of epochs to train")
parser.add_argument(
    "--batch_size", type=int, default=16, help="input batch size for training"
)
parser.add_argument(
    "--split_batch", type=int, default=1, help="split the batch into smaller chunks"
)
parser.add_argument(
    "--test_only", action="store_true", help="set this option to test the model"
)
parser.add_argument("--gan_k", type=int, default=1,
                    help="k value for adversarial loss")

# Optimization specifications
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--decay", type=str, default="200",
                    help="learning rate decay type")
parser.add_argument(
    "--gamma", type=float, default=0.5, help="learning rate decay factor for step decay"
)
parser.add_argument(
    "--optimizer",
    default="ADAM",
    choices=("SGD", "ADAM", "RMSprop"),
    help="optimizer to use (SGD | ADAM | RMSprop)",
)
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--betas", type=tuple,
                    default=(0.9, 0.999), help="ADAM beta")
parser.add_argument(
    "--epsilon", type=float, default=1e-8, help="ADAM epsilon for numerical stability"
)
parser.add_argument("--weight_decay", type=float,
                    default=0, help="weight decay")
parser.add_argument(
    "--gclip",
    type=float,
    default=0,
    help="gradient clipping threshold (0 = no clipping)",
)

# Loss specifications
parser.add_argument(
    "--loss", type=str, default="1*L1", help="loss function configuration"
)
parser.add_argument(
    "--skip_threshold",
    type=float,
    default="1e8",
    help="skipping batch that has large error",
)

# Log specifications
parser.add_argument("--save", type=str, default="edsr_x2",
                    help="file name to save")
parser.add_argument("--load", type=str, default="", help="file name to load")
parser.add_argument(
    "--save_models", action="store_true", help="save all intermediate models"
)
parser.add_argument(
    "--print_every",
    type=int,
    default=100,
    help="how many batches to wait before logging training status",
)
parser.add_argument("--save_results", action="store_true",
                    help="save output results")
parser.add_argument(
    "--save_gt",
    action="store_true",
    help="save low-resolution and high-resolution images together",
)

args = parser.parse_args()

args.scale = list(map(lambda x: int(x), args.scale.split("+")))
args.data_train = args.data_train.split("+")
args.data_test = args.data_test.split("+")

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == "True":
        vars(args)[arg] = True
    elif vars(args)[arg] == "False":
        vars(args)[arg] = False
