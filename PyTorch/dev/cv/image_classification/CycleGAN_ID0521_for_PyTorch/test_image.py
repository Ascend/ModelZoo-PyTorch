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
import random
import timeit

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from cyclegan_pytorch import Generator

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--file", type=str, default="assets/horse.png",
                    help="Image name. (default:`assets/horse.png`)")
parser.add_argument("--model-name", type=str, default="weights/horse2zebra/netG_A2B.pth",
                    help="dataset name.  (default:`weights/horse2zebra/netG_A2B.pth`).")
parser.add_argument("--npu", action="store_true", help="Enables npu")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

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
#device = torch.device("cuda:0" if args.cuda else "cpu")
# Set npu device so everything is done on the right NPU
if torch.npu.is_available() and not args.npu:
    print("WARNING: You have a NPU device, so you should probably run with --npu")
	
device = torch.device("npu:0" if args.npu else "cpu")
################################ modify by npu ##########################################



# create model
model = Generator().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.model_name))

# Set model mode
model.eval()

# Load image
image = Image.open(args.file)
pre_process = transforms.Compose([transforms.Resize(args.image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                  ])
image = pre_process(image).unsqueeze(0)
image = image.to(device)

start = timeit.default_timer()
fake_image = model(image)
elapsed = (timeit.default_timer() - start)
print(f"cost {elapsed:.4f}s")
vutils.save_image(fake_image.detach(), "result.png", normalize=True)
