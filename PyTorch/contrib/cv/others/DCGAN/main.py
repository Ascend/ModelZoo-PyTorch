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
import argparse
import os
import time
import apex
from apex import amp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

from dcgan import Generator, Discriminator, weights_init_normal

parser = argparse.ArgumentParser(description="pytorch DCGAN implementation")
## dcgan parameters
parser.add_argument('--data', metavar='DIR', type=str, default="./data",
                    help='path to dataset')
parser.add_argument("--n-epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch-size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n-cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400,
                    help="interval between image sampling")
## add useful parameters : such as resume,evaluate
parser.add_argument('--checkpoint-path', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model : generate (n_samples) samples,saved in dir(validate)')
parser.add_argument('--n-samples', type=int, default=10,
                    help="amount of samples in function(validate)")
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default 10)')
## parameters for distribute training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
## for ascend 910
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115',
                    type=str, help='master addr')
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7',
                    type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--loss-scale', default=None, type=float,
                    help='loss scale using in amp, default None means dynamic')
parser.add_argument('--opt-level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--prof', default=False, action='store_true',
                    help='use profiling to evaluate the performance of model')


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()

    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def get_device_name(device_type, device_order):
    if device_type == 'npu':
        device_name = 'npu:{}'.format(device_order)
    else:
        device_name = 'cuda:{}'.format(device_order)

    return device_name


def main():
    args = parser.parse_args()
    print(args.device_list)
    args.process_device_map = device_id_to_process_device_map(args.device_list)

    # add start_epoch
    args.start_epoch = 0

    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = '29688'

    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map)
    else:
        if args.gpu is None:
            ngpus_per_node = len(args.process_device_map)
        else:
            ngpus_per_node = 1
    print('ngpus_per_node:', ngpus_per_node)

    args.world_size = ngpus_per_node * args.world_size
    args.distributed = args.world_size > 1

    # create folders
    if not args.distributed or (args.distributed and args.rank == args.process_device_map[0]):
        if not os.path.exists("./images/"):
            os.makedirs("./images/")
        if not os.path.exists("./samples/"):
            os.makedirs("./samples/")

    main_worker(args.rank, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = args.process_device_map[gpu]
    if args.distributed:
        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,
                                    # init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)

    print('rank: {} / {}'.format(args.rank, args.world_size))

    # init device
    device_loc = get_device_name(args.device, args.gpu)
    args.loc = device_loc

    # set device
    print('set_device ', device_loc)
    if args.device == 'npu':
        torch.npu.set_device(device_loc)
    else:
        torch.cuda.set_device(args.gpu)

    # create model
    G = Generator(args.img_size, args.latent_dim, args.channels)
    D = Discriminator(args.img_size, args.channels)
    # initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)
    if args.checkpoint_path:
        print("=> using pre-trained model dcgan,device(%d)" % args.gpu)
        print("loading model of yours...,device(%d)" % args.gpu)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        G.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["G"].items()})
        D.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["D"].items()})
    else:
        print("=> creating model dcgan,device(%d)" % args.gpu)

    print('model to device_loc(%s)...' % device_loc)
    G = G.to(device_loc)
    D = D.to(device_loc)

    if args.distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        args.n_cpu = int((args.n_cpu + ngpus_per_node - 1) / ngpus_per_node)
        args.sample_interval = int(args.sample_interval / ngpus_per_node)

    # define optimizer, apply apex
    optimizer_G = apex.optimizers.NpuFusedAdam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = apex.optimizers.NpuFusedAdam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.amp:
        [D, G], [optimizer_D, optimizer_G] = amp.initialize(
            [D, G], [optimizer_D, optimizer_G], opt_level=args.opt_level, loss_scale=args.loss_scale, num_losses=3,
            combine_grad=True)

    if args.evaluate:
        print("evaluate mode...", " device(%d)," % args.gpu)
        validate(G, args)
        return

    if args.checkpoint_path:
        args.start_epoch = checkpoint['epoch']
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if args.amp:
            amp.load_state_dict(checkpoint['amp'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    if args.distributed:
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[args.gpu], broadcast_buffers=False)
        D = torch.nn.parallel.DistributedDataParallel(D, device_ids=[args.gpu], broadcast_buffers=False)

    # Loss function
    adversarial_loss = nn.BCEWithLogitsLoss().to(device_loc)

    cudnn.benchmark = True

    # Data loading code
    data_path = args.data
    print("dataset path : %s" % data_path)
    train_dataset = datasets.MNIST(
        data_path,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.n_cpu, pin_memory=False, sampler=train_sampler, drop_last=True)

    if args.prof:
        print("profiling mode...", " device(%d)," % args.gpu)
        profiling(train_loader, G, D, optimizer_G, optimizer_D, adversarial_loss, args)
        return

    # start training
    print("train mode...", " device(%d)," % args.gpu)
    fixed_z = torch.randn((5, args.latent_dim), dtype=torch.float32)
    # Configure input
    fixed_z = fixed_z.to(device_loc, non_blocking=True).to(torch.float)
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader,
              G, D,
              optimizer_G, optimizer_D,
              adversarial_loss,
              epoch, args,
              ngpus_per_node)

        if not args.distributed or (args.distributed and args.gpu == args.process_device_map[0]):
            # save fixed imgs
            G.eval()
            fixed_imgs = G(fixed_z)
            save_image(fixed_imgs[:5], "samples/fixed_images-epoch_%03d.png" % epoch, nrow=5, normalize=True)
            ############## npu modify begin #############
            if args.amp:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': 'dcgan',
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'amp': amp.state_dict(),
                }, "checkpoint-amp-epoch_%d.pth" % (epoch + 1))

                if os.path.exists("checkpoint-amp-epoch_%d.pth" % epoch):
                    os.remove("checkpoint-amp-epoch_%d.pth" % epoch)
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': 'dcgan',
                    'G': G.state_dict(),
                    'D': D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                }, "checkpoint-epoch_%d.pth" % (epoch + 1))
                if os.path.exists("checkpoint-epoch_%d.pth" % epoch):
                    os.remove("checkpoint-epoch_%d.pth" % epoch)
            ############## npu modify end #############
    # train loop done


def profiling(train_loader, generator, discriminator, optimizer_G, optimizer_D, loss, args):
    generator.train()
    discriminator.train()

    def update(step=None):
        start_time = time.time()
        valid = torch.ones(imgs.size(0), 1, requires_grad=False)
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False)
        # Sample noise as generator input
        z = torch.randn((imgs.size(0), args.latent_dim), dtype=torch.float32)
        # Configure input
        real_imgs = imgs.to(args.loc, non_blocking=True).to(torch.float)
        valid = valid.to(args.loc, non_blocking=True).to(torch.float)
        fake = fake.to(args.loc, non_blocking=True).to(torch.float)
        z = z.to(args.loc, non_blocking=True).to(torch.float)
        # update D
        discriminator.zero_grad()
        output = discriminator(real_imgs)
        errD_real = loss(output, valid)
        with amp.scale_loss(errD_real, optimizer_D, loss_id=0) as errD_real_scaled:
            errD_real_scaled.backward()
        gen_imgs = generator(z)
        output = discriminator(gen_imgs.detach())
        errD_fake = loss(output, fake)
        with amp.scale_loss(errD_fake, optimizer_D, loss_id=1) as errD_fake_scaled:
            errD_fake_scaled.backward()
        errD = errD_real + errD_fake
        optimizer_D.step()
        # update G
        generator.zero_grad()
        output = discriminator(gen_imgs)
        errG = loss(output, valid)
        with amp.scale_loss(errG, optimizer_G, loss_id=2) as errG_scaled:
            errG_scaled.backward()
        optimizer_G.step()
        if step is not None:
            print('iter: %d, loss: %.2f, time: %.2f' % (step, errG.item(), (time.time() - start_time)))

    for i, (imgs, _) in enumerate(train_loader):
        if i < 20:
            update(step=i)
        else:
            if args.device == 'npu':
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    update()
            else:
                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    update()
            break
    prof.export_chrome_trace("dcgan.prof")


def train(train_loader, generator, discriminator, optimizer_G, optimizer_D, loss, epoch, args, ngpus_per_node):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    G_loss = AverageMeter('G_Loss', ':.4e')
    D_loss = AverageMeter('D_Loss', ':.4e')
    D_real = AverageMeter('D_real', ':.4e')
    D_fake = AverageMeter('D_fake', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, G_loss, D_loss, D_real, D_fake],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    generator.train()
    discriminator.train()

    end = time.time()
    for i, (imgs, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        valid = torch.ones(imgs.size(0), 1, requires_grad=False)
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False)
        # Sample noise as generator input
        z = torch.randn((imgs.size(0), args.latent_dim), dtype=torch.float32)
        # Configure input
        real_imgs = imgs.to(args.loc, non_blocking=True).to(torch.float)
        valid = valid.to(args.loc, non_blocking=True).to(torch.float)
        fake = fake.to(args.loc, non_blocking=True).to(torch.float)
        z = z.to(args.loc, non_blocking=True).to(torch.float)

        # update D
        discriminator.zero_grad()
        output = discriminator(real_imgs)
        errD_real = loss(output, valid)
        with amp.scale_loss(errD_real, optimizer_D, loss_id=0) as errD_real_scaled:
            errD_real_scaled.backward()

        gen_imgs = generator(z)
        output = discriminator(gen_imgs.detach())
        errD_fake = loss(output, fake)
        with amp.scale_loss(errD_fake, optimizer_D, loss_id=1) as errD_fake_scaled:
            errD_fake_scaled.backward()
        errD = errD_real + errD_fake
        optimizer_D.step()

        # update G
        generator.zero_grad()
        output = discriminator(gen_imgs)
        errG = loss(output, valid)
        with amp.scale_loss(errG, optimizer_G, loss_id=2) as errG_scaled:
            errG_scaled.backward()
        optimizer_G.step()

        D_loss.update(errD.item(), real_imgs.size(0))
        D_fake.update(errD_fake.item(), real_imgs.size(0))
        D_real.update(errD_real.item(), real_imgs.size(0))
        G_loss.update(errG.item(), real_imgs.size(0))

        # measure elapsed time
        cost_time = time.time() - end
        batch_time.update(cost_time)
        end = time.time()

        if not args.distributed or (args.distributed and args.gpu == args.process_device_map[0]):
            if i % args.print_freq == 0:
                progress.display(i)

            batches_done = epoch * len(train_loader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%06d.png" % batches_done, nrow=5, normalize=True)

            if batch_time.avg:
                print("[npu id:", args.gpu, "]", "batch_size:", args.world_size * args.batch_size,
                      'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                        args.batch_size * args.world_size / batch_time.avg))
    # train loop done


def validate(generator, args):
    batch_time = AverageMeter('Time', ':6.3f')
    print("start generate random image...(validate mode)")
    generator.eval()

    if not os.path.exists("./validate/"):
        os.makedirs("validate")
    end = time.time()
    with torch.no_grad():
        for i in range(args.n_samples):
            z = torch.randn((25, args.latent_dim), dtype=torch.float32)
            z = z.to(args.loc, non_blocking=True)
            # gen images
            images = generator(z)
            batch_time.update(time.time() - end)
            end = time.time()
            save_image(images.data[:25], "validate/%03d.jpg" % i, nrow=5, normalize=True)
            if batch_time.avg:
                print("[npu id:", args.gpu, "]", "batch_size:", 25,
                      'Time: {:.3f}'.format(batch_time.avg), '* FPS@all {:.3f}'.format(
                        25 / batch_time.avg))
        # train loop done


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=2):
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


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()
