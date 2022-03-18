# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="horse2zebra",
                    help="dataset name. (default:`horse2zebra`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, "
                         "iphone2dslr_flower, ae_photos, ]")
parser.add_argument("--npu", action="store_true", help="Enables npu")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default: `./results`).")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

################################ modify by npu ##########################################
# Set cuda device so everything is done on the right GPU
#if torch.cuda.is_available() and not args.cuda:
#    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# Set npu device so everything is done on the right NPU
if torch.npu.is_available() and not args.npu:
    print("WARNING: You have a NPU device, so you should probably run with --npu")
################################ modify by npu ##########################################

# Dataset
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize(args.image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ]),
                       mode="test")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

try:
    os.makedirs(os.path.join(args.outf, str(args.dataset), "A"))
    os.makedirs(os.path.join(args.outf, str(args.dataset), "B"))
except OSError:
    pass

################################ modify by npu ##########################################
# Set cuda device so everything is done on the right GPU
#device = torch.device("cuda:0" if args.cuda else "cpu")
# Set npu device so everything is done on the right NPU
device = torch.device("npu:0" if args.npu else "cpu")
################################ modify by npu ##########################################

# create model
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(os.path.join("weights", str(args.dataset), "netG_A2B.pth")))
netG_B2A.load_state_dict(torch.load(os.path.join("weights", str(args.dataset), "netG_B2A.pth")))

# Set model mode
netG_A2B.eval()
netG_B2A.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    real_images_A = data["A"].to(device)
    real_images_B = data["B"].to(device)

    # Generate output
    fake_image_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
    fake_image_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

    # Save image files
    vutils.save_image(fake_image_A.detach(), f"{args.outf}/{args.dataset}/A/{i + 1:04d}.png", normalize=True)
    vutils.save_image(fake_image_B.detach(), f"{args.outf}/{args.dataset}/B/{i + 1:04d}.png", normalize=True)

    progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")
