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
import itertools
import os
import random
import time
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import apex
from apex import amp
from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import ReplayBuffer
from cyclegan_pytorch import weights_init

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:./data)")
parser.add_argument("--dataset", type=str, default="horse2zebra",
                    help="dataset name. (default:`horse2zebra`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, maps, facades, selfie2anime, "
                         "iphone2dslr_flower, ae_photos, ]")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--decay_epochs", type=int, default=100,
                    help="epoch to start linearly decaying the learning rate to 0. (default:100)")
parser.add_argument("-b", "--batch-size", default=1, type=int,
                    metavar="N",
                    help="mini-batch size (default: 1), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate. (default:0.0002)")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="print frequency. (default:100)")
parser.add_argument("--npu", action="store_true", help="Enables npu")
parser.add_argument("--netG_A2B", default="", help="path to netG_A2B (to continue training)")
parser.add_argument("--netG_B2A", default="", help="path to netG_B2A (to continue training)")
parser.add_argument("--netD_A", default="", help="path to netD_A (to continue training)")
parser.add_argument("--netD_B", default="", help="path to netD_B (to continue training)")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--outf", default="",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

parser.add_argument('--apex', action='store_true',help='Use apex for mixed precision training')
parser.add_argument('--apex-opt-level', default='O2', type=str,
                    help='For apex mixed precision training'
                         'O0 for FP32 training, O1 for mixed precision training.')
parser.add_argument('--loss-scale-value', default=128., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--max_steps', default=None, type=int, metavar='N',
                        help='number of total steps to run')



args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
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
                           transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                           transforms.RandomCrop(args.image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=128, pin_memory=True)

try:
    os.makedirs(os.path.join(args.outf, args.dataset, "A"))
    os.makedirs(os.path.join(args.outf, args.dataset, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", args.dataset))
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
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

if args.netG_A2B != "":
    netG_A2B.load_state_dict(torch.load(args.netG_A2B))
if args.netG_B2A != "":
    netG_B2A.load_state_dict(torch.load(args.netG_B2A))
if args.netD_A != "":
    netD_A.load_state_dict(torch.load(args.netD_A))
if args.netD_B != "":
    netD_B.load_state_dict(torch.load(args.netD_B))

# define loss function (adversarial_loss) and optimizer
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optimizer_G = apex.optimizers.NpuFusedAdam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_D_A = apex.optimizers.NpuFusedAdam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_B = apex.optimizers.NpuFusedAdam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))


if args.apex:
    [netG_A2B, netG_B2A, netD_A, netD_B],[optimizer_G, optimizer_D_A, optimizer_D_B] = amp.initialize(
                                         [netG_A2B, netG_B2A, netD_A, netD_B],[optimizer_G, optimizer_D_A, optimizer_D_B],
                                         opt_level=args.apex_opt_level,
                                         loss_scale=args.loss_scale_value,
                                         combine_grad=True)

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

g_losses = []
d_losses = []

identity_losses = []
gan_losses = []
cycle_losses = []

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        if args.max_steps and i > args.max_steps:
            break
        start_time = time.time()
        # get batch size data
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)
        batch_size = real_image_A.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = netG_B2A(real_image_A)
        loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = netG_A2B(real_image_B)
        loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = netG_B2A(real_image_B)
        fake_output_A = netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = netG_A2B(real_image_A)
        fake_output_B = netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

        recovered_image_B = netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Calculate gradients for G_A and G_B
        #errG.backward()
        if args.apex:
            with amp.scale_loss(errG,optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = netD_A(real_image_A)
        errD_real_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_output_A = netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        #errD_A.backward()
        if args.apex:
            with amp.scale_loss(errD_A,optimizer_D_A) as scaled_loss:
                scaled_loss.backward()
        else:
            errD_A.backward()
        # Update D_A weights
        optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = netD_B(real_image_B)
        errD_real_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        #errD_B.backward()
        if args.apex:
            with amp.scale_loss(errD_B,optimizer_D_B) as scaled_loss:
                scaled_loss.backward()
        else:
            errD_B.backward()



        # Update D_B weights
        optimizer_D_B.step()

        #
        step_time = time.time() - start_time
        fps = args.batch_size / step_time

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(errD_A + errD_B).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f} "
            f"step_time: {step_time:.4f} "
            f"fps: {fps:.4f}")

        if i % args.print_freq == 0:
            vutils.save_image(real_image_A,
                              f"{args.outf}/{args.dataset}/A/real_samples.png",
                              normalize=True)
            vutils.save_image(real_image_B,
                              f"{args.outf}/{args.dataset}/B/real_samples.png",
                              normalize=True)

            fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{args.outf}/{args.dataset}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{args.outf}/{args.dataset}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    # do check pointing
    torch.save(netG_A2B.state_dict(), f"weights/{args.dataset}/netG_A2B_epoch_{epoch}.pth")
    torch.save(netG_B2A.state_dict(), f"weights/{args.dataset}/netG_B2A_epoch_{epoch}.pth")
    torch.save(netD_A.state_dict(), f"weights/{args.dataset}/netD_A_epoch_{epoch}.pth")
    torch.save(netD_B.state_dict(), f"weights/{args.dataset}/netD_B_epoch_{epoch}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

# save last check pointing
torch.save(netG_A2B.state_dict(), f"weights/{args.dataset}/netG_A2B.pth")
torch.save(netG_B2A.state_dict(), f"weights/{args.dataset}/netG_B2A.pth")
torch.save(netD_A.state_dict(), f"weights/{args.dataset}/netD_A.pth")
torch.save(netD_B.state_dict(), f"weights/{args.dataset}/netD_B.pth")
