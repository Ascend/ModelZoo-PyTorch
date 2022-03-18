#
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
#

import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net
import torch.npu
import os



print('aaaaaaaaaaaaaaaaaaaa')
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()

# device
device = "npu:{}".format(args.gpu_id) if torch.npu.is_available() and not args.no_cuda else "cpu"
if torch.npu.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader_sampler = torch.utils.data.distributed.DistributedSampler(torchvision.datasets.ImageFolder(query_dir, transform=transform))
queryloader_batch_size = int(64 / int(os.getenv('NPU_WORLD_SIZE')))
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=queryloader_batch_size, shuffle=False, 
pin_memory = False, drop_last = True, sampler = queryloader_sampler)
galleryloader_sampler = torch.utils.data.distributed.DistributedSampler(torchvision.datasets.ImageFolder(gallery_dir, transform=transform))
galleryloader_batch_size = int(64 / int(os.getenv('NPU_WORLD_SIZE')))
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=galleryloader_batch_size, shuffle=False, 
pin_memory = False, drop_last = True, sampler = galleryloader_sampler)

# net definition
net = Net(reid=True)
net = net.to(f'npu:{NPU_CALCULATE_DEVICE}')
if not isinstance(net, torch.nn.parallel.DistributedDataParallel):
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[NPU_CALCULATE_DEVICE], broadcast_buffers=False)
assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.t7')
checkpoint = torch.load("./checkpoint/ckpt.t7")
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(f'npu:{NPU_CALCULATE_DEVICE}')

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx,(inputs,labels) in enumerate(queryloader):
        inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))

    for idx,(inputs,labels) in enumerate(galleryloader):
        inputs = inputs.to(f'npu:{NPU_CALCULATE_DEVICE}')
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels -= 2

# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
}
torch.save(features,"features.pth")
