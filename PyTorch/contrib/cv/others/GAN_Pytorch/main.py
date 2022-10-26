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
# ============================================================================
import torch
if torch.__version__ >= "1.8":
    try:
        import torch_npu
    except:
        print('WARNING! torch_npu is not imported.. Please using without npu..')         
import argparse
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from models import Generator, Discriminator

try:
    import apex
    from apex import amp
except ImportError:
    amp = None

def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print

print = flush_print(print)


def train_one_epoch(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss,
                        epoch, args, dataloader, Tensor,LOSS_G,LOSS_D,device):
    batch_time = AverageMeter('Time', ':6.3f', start_count_index=5)
    G_loss = AverageMeter('g_loss', ':6.3f', start_count_index=0)
    D_loss = AverageMeter('d_loss', ':6. 3f', start_count_index=0)

    for i, (imgs,_) in enumerate(dataloader):

        start_time = time.time()
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(torch.Tensor)).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))).to(device)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        G_loss.update(g_loss.item(), len(gen_imgs))
        if args.apex:
            with amp.scale_loss(g_loss, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            G_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        D_loss.update(d_loss.item(), len(real_imgs))
        if args.apex:
            with amp.scale_loss(d_loss, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            d_loss.backward()
        optimizer_D.step()
        batch_time.update(time.time() - start_time)
        if args.n_epochs == 1 and args.is_master_node:
            print(
                "[Epoch %d] [step %d] [D loss: %f] [G loss: %f]"
                % (epoch, i, D_loss.avg, G_loss.avg)
            )
        batches_done = epoch * len(dataloader)+ i
        if batches_done % args.sample_interval == 0 and args.is_master_node and args.n_epochs != 1:
            save_image(gen_imgs.data[:25], "training_images/%d.png" % batches_done, nrow=5, normalize=True)
    if args.is_master_node:
        print(
            "[Epoch %d] [D loss: %f] [G loss: %f] FPS:%.3f"
            % (epoch, D_loss.avg, G_loss.avg, args.batch_size * args.gpus / batch_time.avg)
        )
    LOSS_G.append(G_loss.avg)
    LOSS_D.append(D_loss.avg)



def main(args):

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'

    if args.apex:
        if sys.version_info < (3, 0):
            raise RuntimeError("Apex currently only supports Python 3. Aborting.")
        if amp is None:
            raise RuntimeError("Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                               "to enable mixed-precision training.")
    
    device = torch.device(f'npu:{args.local_rank}')  # npu
    torch.npu.set_device(f'npu:{args.local_rank}')
    print('device_id=', args.local_rank)
    if args.distributed:
        torch.distributed.init_process_group(backend='hccl', world_size=args.gpus, rank=args.local_rank)

    args.is_master_node = not args.distributed or args.local_rank == 0

    if args.is_master_node:
        print(args)
        print("Preparing dataset...")

    # Configure data loader
    train_dataset = datasets.MNIST(
        args.data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ))
    
    if args.is_master_node:    
        print("Creating dataloader")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.is_master_node:
        print("Creating model")

    Tensor = torch.npu.FloatTensor
    LOSS_G=[]
    LOSS_D=[]

    generator = Generator()
    discriminator = Discriminator()
    if args.pretrained:
        print("=> using pre-trained model GAN")
        generator = Generator()
        discriminator = Discriminator()
        print("loading model of yours...")
        checkpoint = torch.load(r'./checkpoint.pth.tar',map_location='cpu')
        from collections import OrderedDict
        new_state_dict_g = OrderedDict()
        new_state_dict_d = OrderedDict()
        for k, v in checkpoint['state_dict_G'].items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict_g[name] = v
        for k, v in checkpoint['state_dict_D'].items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict_d[name] = v
        # load params
        generator.load_state_dict(new_state_dict_g)
        discriminator.load_state_dict(new_state_dict_d)
        LOSS_D = checkpoint['loss_d']
        LOSS_G = checkpoint['loss_g']
        args.start_epoch = checkpoint['epoch']

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    adversarial_loss = nn.BCELoss().to(device)

    optimizer_G = apex.optimizers.NpuFusedAdam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = apex.optimizers.NpuFusedAdam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.apex:
        amp.register_float_function(torch, 'sigmoid')
        amp.register_half_function(torch, 'addmm')
        generator, optimizer_G = amp.initialize(generator, optimizer_G,
                                                opt_level='O1', loss_scale="dynamic",combine_grad=True)

        discriminator, optimizer_D = amp.initialize(discriminator, optimizer_D,
                                                    opt_level='O1', loss_scale="dynamic",combine_grad=True)

    if args.distributed:
        generator = DDP(generator, device_ids=[args.local_rank], broadcast_buffers=False)
        discriminator = DDP(discriminator, device_ids=[args.local_rank], broadcast_buffers=False)

    if args.test_only :
        os.makedirs("test_images",exist_ok=True)
        Tensor = torch.npu.FloatTensor
        generator = Generator().npu()
        checkpoint = torch.load(r'./checkpoint.pth.tar', map_location='cpu')

        loss_d = checkpoint['loss_d']
        loss_g = checkpoint['loss_g']
        x = range(len(loss_d))
        plt.figure()

        plt.plot(x, loss_d, color='r', label="loss_d")
        plt.plot(x, loss_g, color='g', label="loss_g")
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.savefig('LOSS_{}p_{}_{}.jpg'.format(args.gpus, args.lr, args.batch_size))

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict_G'].items():
            name = k.replace("module.", "")  # remove `module.`
            new_state_dict[name] = v
        # load params
        generator.load_state_dict(new_state_dict)
        os.makedirs("image", exist_ok=True)
        for i in range(200):
            z = Variable(Tensor(np.random.normal(0, 1, (64, 100)))).npu()

            # Generate a batch of images
            gen_imgs = generator(z)

            save_image(gen_imgs.data[:25], "test_images/image/%d.png" % i, nrow=5, normalize=True)
        print("Generate done!")
        return

    if args.is_master_node:
        print("Start training")
    start_time = time.time()
    os.makedirs("training_images",exist_ok=True)
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(generator, discriminator, optimizer_G, optimizer_D, adversarial_loss,
                        epoch, args, dataloader,Tensor, LOSS_G,LOSS_D,device)

        if epoch == 50 or epoch == 199:
            if args.apex and args.is_master_node:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'GAN',
                    'state_dict_G': generator.state_dict(),
                    'state_dict_D': discriminator.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'loss_g': LOSS_G,
                    'loss_d': LOSS_D,
                    'apex': amp.state_dict()
                })
            elif args.is_master_node:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'GAN',
                    'state_dict_G': generator.state_dict(),
                    'state_dict_D': discriminator.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'loss_g': LOSS_G,
                    'loss_d': LOSS_D
                })
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.is_master_node:
        print('Training time {}'.format(total_time_str))

def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=10):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GAN Training')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0008, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--gpus", type=int, default=1, help="num of gpus of per node")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument('--local_rank', default=0, type=int, help='device id')
    parser.add_argument("--test_only", type=int, default=None, help="only generate images")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    # 数据集path
    parser.add_argument('--data_path', default='../data/mnist',
                        help='the path of the dataset')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ## for ascend 910
    parser.add_argument('--addr', default='127.0.0.1',
                        type=str, help='master addr')
    parser.add_argument('--workers', default=16, type=int,
                        help='numbers of worker')
    parser.add_argument('--apex', default=False, action='store_true',
                        help='use apex to train the model')
    args = parser.parse_args()
    
    args.gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    
    return args
if __name__ == '__main__':
    args = parse_args()
    main(args)