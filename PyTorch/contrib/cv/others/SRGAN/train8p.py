# coding:GBK
# Copyright 2020 Huawei Technologies Co., Ltd
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
import random
import time
import warnings
import tempfile
import math

import torch
if torch.__version__>= "1.8":
    #print("import torch_npu")
    import torch_npu
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
import torchvision.utils as utils

from apex import amp

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model import Generator, Discriminator
from loss import GeneratorLoss
import pytorch_ssim
import config
from config import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--crop_size', default=88, type=int,
                        help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--train_data_path', default='./data/VOC2012/train', type=str,
                    help='source data folder for training')
parser.add_argument('--val_data_path', default='./data/VOC2012/val', type=str,
                    help='source data folder for training')
parser.add_argument('--only_keep_best', default=True, type=bool,
                    help='If use gpu for training.')
parser.add_argument('--performance', default=False, type=bool,
                        help='If run val process.')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--amp', default=False, action='store_true', help='use amp to train the model')
parser.add_argument('--loss_scale_g', default=128.0, help='netG amp loss_scale: dynamic, 128.0')
parser.add_argument('--loss_scale_d', default=128.0, help='netD amp loss_scale: dynamic, 128.0')
parser.add_argument('--amp_level', default='O1', type=str,
                    help='loss scale using in amp, default -1 means dynamic')

parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--workspace', type=str, default='./', metavar='DIR',
                    help='path to directory where checkpoints will be stored')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-bm', '--benchmark', default=0, type=int,
                    metavar='N', help='set benchmark status (default: 1,run benchmark)')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
parser.add_argument('--device_num', default=-1, type=int,
                    help='device_num')
parser.add_argument('--output_dir', default=config.get_root_path(), type=str,
                    help='Path to save running results.')
warnings.filterwarnings('ignore')
best_acc1 = 0
config.set_root_path(parser.parse_args().output_dir)

def main():
    args = parser.parse_args()
    print("===============main()=================")
    print(args)
    print("===============main()=================")

    os.environ['LOCAL_DEVICE_ID'] = str(0)
    print("+++++++++++++++++++++++++++LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29688'

    # 结果保存路径

    if not os.path.exists(config.get_root_path() + 'epochs'):
        os.makedirs(config.get_root_path() + 'epochs')

    print(config.get_root_path())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.device_list != '':
        ngpus_per_node = len(args.device_list.split(','))
    elif args.device_num != -1:
        ngpus_per_node = args.device_num
    elif args.device == 'npu':
        ngpus_per_node = int(os.environ["RANK_SIZE"])
    else:
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        print('ngpus_per_node:', ngpus_per_node)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # 计时操作
    avt = AverageMeter(args.performance)
    if args.device_list != '':
        args.gpu = int(args.device_list.split(',')[gpu])
    else:
        args.gpu = gpu
    # 在主线程中打印
    if args.rank == 0:
        print("[npu id:", args.gpu, "]", "++++++++++++++++ before set LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])
        os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
        print("[npu id:", args.gpu, "]", "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID'])

    if args.gpu is not None:
        print("[npu id:", args.gpu, "]", "Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        if args.device == 'npu':
            print(f'****** Init process,current rank is:{args.rank}********')
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)

    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    if args.rank == 0:
        print("[npu id:", args.gpu, "]", "===============main_worker()=================")
        print("[npu id:", args.gpu, "]", args)
        print("[npu id:", args.gpu, "]", "===============main_worker()=================")

    # load data
    # 实例化训练数据集
    train_data_set = TrainDatasetFromFolder(args.train_data_path, args.crop_size, args.upscale_factor)
    val_data_set = ValDatasetFromFolder(args.val_data_path, args.upscale_factor)

    # 给每个rank对应的进程分配训练的样本索引, （这一步相当于把整个数据集直接分成了GPU数量份）
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=1,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=args.workers)

    # create models
    netG = Generator(args.upscale_factor).to(loc)  # 这里会将网络设置到不同的GPU上
    netD = Discriminator().to(loc)
    # 这一步操作是为了同步各个GPU上的权重，权重不同会导致训练结果有误
    netG_checkpoint_path = os.path.join(tempfile.gettempdir(), "NetG_initial_weights.pt")
    netD_checkpoint_path = os.path.join(tempfile.gettempdir(), "NetD_initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保持， 然后其他进程载入，保持初始化权重一致
    if args.gpu == 0:
        torch.save(netG.state_dict(), netG_checkpoint_path)
        torch.save(netD.state_dict(), netD_checkpoint_path)

    # dist.barrier()  # 等待所有GPU处理完
    torch.npu.synchronize()

    # define loss function (criterion) and optimizer
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    if args.amp:
        netG, optimizerG = amp.initialize(netG, optimizerG, opt_level=args.amp_level, loss_scale=args.loss_scale_g)
        netD, optimizerD = amp.initialize(netD, optimizerD, opt_level=args.amp_level, loss_scale=args.loss_scale_d)

    # 转为DDP模型
    print(f'Converting model to DDP model...')
    netG = torch.nn.parallel.DistributedDataParallel(
        netG, device_ids=[args.gpu], output_device=args.gpu, broadcast_buffers=False)
    netD = torch.nn.parallel.DistributedDataParallel(
        netD, device_ids=[args.gpu], output_device=args.gpu, broadcast_buffers=False)

    cudnn.benchmark = True

    results = {'epoch': [], 'd_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [],
               'train_fps': []}

    for epoch in range(1, args.epochs+1):
        avt.t_start('epoch')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # if args.rank == 0:
        print(f'{"#"*10}-device:{loc} start train epoch:{epoch}-{"#"*10}')
        # train for one epoch
        running_results = train_one_epoch(netG, netD, optimizerG, optimizerD, train_loader, loc, epoch, args.rank, avt,
                                          amp_b=args.amp, number_epoch=args.epochs, world_size=ngpus_per_node)
        if args.rank == 0:
            avt.print_time('training')
        # validate
        # if args.rank == 0:
        if not args.performance:
            print(f'{"#"*10}-device:{loc} start validate epoch:{epoch}-{"#"*10}')
            avt.t_start('val')
            valing_results = evaluate(netG, epoch, val_loader, loc, args.rank)
            if args.rank == 0:
                avt.print_time('val')
        else:
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        # saving results
        if args.rank == 0:
            epoch_results_save(netG, netD, args.performance, running_results, valing_results,
                               epoch, results, args.only_keep_best)
            avt.print_time('epoch')
    if args.rank == 0:
        save_training_log(results)
        #dist.destroy_process_group()
        avt.print_time('end')


def train_one_epoch(netG, netD, optimizerG, optimizerD, data_loader, device, epoch, rank, avt,
                    amp_b=False, number_epoch=100, world_size=1):
    # 记录单个epoch的运行结果
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'train_fps': 0}
    # 进入训练模式
    netG.train()
    netD.train()

    # 创建损失
    generator_criterion = GeneratorLoss()
    # 实例化生成器损失函数模型
    if torch.npu.is_available():
        generator_criterion = generator_criterion.to(device)
    # mean_loss = torch.zeros(1).to(device)  # 定义一个平均损失的变量，初始化为0

    # fps 统计方法
    fps_number = 0
    fps_count_start = False

    avt.t_start('training')
    for step, data in enumerate(data_loader):
        #print(f'当前步骤为:{step}----------------------------------------')
        images, labels = data  # labels(batchsize*3*44*44)
        batch_size = images.size(0)
        running_results['batch_sizes'] += batch_size

        fps_start_time = time.time()
        if step == 5:
            fps_count_start = True
        #########################
        # (1) Update D network:  maximize D(x) -1 - D(G(Z))
        #########################
        real_img = Variable(labels)
        real_img = real_img.to(device)
        z = Variable(images)
        z = z.to(device)
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out

        if amp_b:
            with amp.scale_loss(d_loss, optimizerD) as scaled_d_loss:
                scaled_d_loss.backward(retain_graph=True)
        else:
            d_loss.backward(retain_graph=True)

        # # 多GPU同步d_loss
        # d_loss = reduce_value(d_loss, average=True)
        stream = torch.npu.current_stream()
        stream.synchronize()

        optimizerD.step()

        #######################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        #######################
        netG.zero_grad()
        ## The two lines below are added to prevent runetime error in Google Colab ##
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()
        ##
        # generator_criterion.to(device)  #
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        if amp_b:
            with amp.scale_loss(g_loss, optimizerG) as scaled_g_loss:
                scaled_g_loss.backward()
        else:
            g_loss.backward()

        # # 多GPU同步g_loss
        # g_loss = reduce_value(g_loss, average=True)
        stream = torch.npu.current_stream()
        stream.synchronize()

        #fake_img = netG(z)
        #fake_out = netD(fake_img).mean()

        # g_loss = reduce_value(g_loss,average=True)
        optimizerG.step()

        # 统计 fps
        if fps_count_start:
            fps = batch_size * world_size / (time.time() - fps_start_time)
            fps = round(fps, 2)
            fps_number += 1
            running_results['train_fps'] += fps
        else:
            fps = 0
        # # 多GPU同步d_score, g_score
        # real_out = reduce_value(real_out, average=True)
        # fake_out = reduce_value(fake_out, average=True)
        # fps = torch.Tensor(fps)
        # fps = reduce_value(fps, average=False)

        # loss for current batch before optimization
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        # 在进程 0 中打印平均loss
        if rank == 0:
            # 如果是主进程，
            Loss_D = running_results['d_loss'] / running_results['batch_sizes']
            Loss_G = running_results['g_loss'] / running_results['batch_sizes']
            score_D = running_results['d_score'] / running_results['batch_sizes']
            score_G = running_results['g_score'] / running_results['batch_sizes']
            print(f'[{epoch}/{number_epoch}] step:{step} Loss_D: {Loss_D:.4f} Loss_G: {Loss_G:.4f} '
                  f'D(x): {score_D:.4f} D(G(z)): {score_G:.4f} Fps: {fps:.4f}')
        avt.step_update()
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.npu.synchronize(device)

    running_results['train_fps'] = running_results['train_fps']/fps_number
    running_results['train_fps'] = round(running_results['train_fps'], 2)

    return running_results

@torch.no_grad()
def evaluate(netG, epoch, data_loader, device, rank, save_val_img=False):
    netG.eval()
    out_path = config.get_root_path() + 'training_val_img/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []

    for step, data in enumerate(data_loader):
        val_lr, val_hr_restore, val_hr = data
        batch_size = val_hr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = val_lr
        hr = val_hr
        lr = lr.to(device)
        hr = hr.to(device)
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).item()
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * math.log10(
            (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

        if rank == 0:
            if step % 10 == 0 or step == 0:
                psnr = valing_results['psnr']
                ssim = valing_results['ssim']
                print(f'[converting LR images to SR images] PSNR: {psnr:4f} dB SSIM: {ssim:4f}')
        if save_val_img:
            val_images.append(display_transform()(val_hr_restore.squeeze(0)))
            val_images.append(display_transform()(hr.data.cpu().squeeze(0)))
            val_images.append(display_transform()(sr.data.cpu().squeeze(0)))
    if save_val_img:
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        if rank == 0:
            index = 1
            for image in val_images:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.npu.synchronize(device)

    return valing_results


def epoch_results_save(netG, netD, performance, running_results, valing_results, epoch, results, only_best):
    global best_acc1
    # 保存每次运行结果， psnr（峰值信噪比） ssim（结构相似性）
    results['epoch'].append(str(epoch))
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['train_fps'].append(running_results['train_fps'])
    if not performance:
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    else:
        results['psnr'].append(0)
        results['ssim'].append(0)
    # print(f"当前数据长度为:{len(results['d_loss'])}")
    save_training_log(results)
    # if epoch % 10 == 0 and epoch != 0:
    #     # save model parameters
    if only_best and not performance:
        current_val = valing_results['psnr']/10 + valing_results['ssim']
        if current_val > best_acc1:
            print(f'current best validation results -> psnr: {valing_results["psnr"]}, '
                  f'ssim: {valing_results["ssim"]}')
            print(config.get_root_path() + 'epochs/netG_best.pth')
            torch.save(netG.module.state_dict(), config.get_root_path() + 'epochs/netG_best.pth')
            torch.save(netD.module.state_dict(), config.get_root_path() + 'epochs/netD_best.pth')
            best_acc1 = current_val
    else:
        if epoch == 1 or epoch % 5 == 0:
            torch.save(netG.module.state_dict(), config.get_root_path() + 'epochs/netG_epoch_%d.pth' % (epoch))
            torch.save(netD.module.state_dict(), config.get_root_path() + 'epochs/netD_epoch_%d.pth' % (epoch))
    return


def save_training_log(results):
    with open(config.get_root_path() + 'epoch_log_8p.txt', 'w', encoding='utf-8') as f:
        title = 'epoch \t d_loss \t g_loss \t d_score \t g_score \t psnr \t ssim \t train_fps \n'
        f.write(title)
        for i in range(len(results['epoch'])):
            str = f"{i + 1} \t {results['d_loss'][i]:.4f} \t {results['g_loss'][i]:.4f} \t " \
                  f"{results['d_score'][i]:.4f} \t {results['g_score'][i]:.4f} \t " \
                  f"{results['psnr'][i]:.4f} \t " \
                  f"{results['ssim'][i]:.4f} \t {results['train_fps'][i]:.4f} \n"
            f.write(str)
    print('write results successfully!')
    return


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(5)
    main()
