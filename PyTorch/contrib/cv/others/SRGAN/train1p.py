# -*- coding: utf-8 -*-
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

import argparse
import os
from math import log10
import time
import torch
if torch.__version__>= "1.8.1":
    print("import torch_npu")
    import torch_npu

import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
import config
from config import AverageMeter

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=64, type=int, help='set training batch size')
parser.add_argument('--amp', default=False, type=bool,
                    help='use amp to train the model')
parser.add_argument('--amp_level', default='O1',
                    help='set amp level.')
parser.add_argument('--train_data_path', default='./data/VOC2012/train', type=str,
                    help='source data folder for training')
parser.add_argument('--val_data_path', default='./data/VOC2012/val', type=str,
                    help='source data folder for training')
parser.add_argument('--device', default='npu',
                        help='device id (i.e. npu:1 or 1,2 or cpu)')
parser.add_argument('--use_npu', default=False, type=bool,
                    help='If use npu for training.')
parser.add_argument('--use_gpu', default=False, type=bool,
                    help='If use gpu for training.')
parser.add_argument('--only_keep_best', default=True, type=bool,
                    help='Only keep best training result.')
parser.add_argument('--save_prof', default=True, type=bool,
                    help='If save training prof.')
parser.add_argument('--loss_scale_g', default=128.0, help='netG amp loss_scale: dynamic, 128.0')
parser.add_argument('--loss_scale_d', default=128.0, help='netD amp loss_scale: dynamic, 128.0')
parser.add_argument('--nproc', default=8, type=int, help='workers for dataset_loader')
parser.add_argument('--save_val_img', default=False, type=bool,
                    help='save val_images')
parser.add_argument('--performance', default=False, type=bool,
                    help='If run val process.')
parser.add_argument('--output_dir', default=config.get_root_path(), type=str,
                    help='Path to save running results.')


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    seed_everything(5)
    # 计时操作
    avt = AverageMeter(opt.performance)

    # 选择训练设备
    if opt.use_npu:
        import torch.npu
        if torch.npu.is_available():
            device = torch.device(opt.device)
            prof_kwargs = {'use_npu': True}
    elif opt.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            prof_kwargs = {'use_cuda': True}
    else:
        device = torch.device('cpu')
        prof_kwargs = {}

    print(f'使用 {device} 进行训练.')

    # 初始化参数
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # 结果保存路径
    config.set_root_path(opt.output_dir)
    out_path = config.get_root_path() + 'epochs'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if opt.save_val_img:
        val_path = config.get_root_path() + 'val_results_img'
        if not os.path.exists(val_path):
            os.makedirs(val_path)

    train_set = TrainDatasetFromFolder(opt.train_data_path, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(opt.val_data_path, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.nproc,
                              batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.nproc, batch_size=1, shuffle=False)

    # 创建生成器实例 netG, 输出生成器参数的数量
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # 实例化生成器损失函数模型
    generator_criterion = GeneratorLoss()

    # 如果能GPU加速，把网络放到GPU上
    if opt.use_npu or opt.use_gpu:
        netG = netG.to(device)
        netD = netD.to(device)
        generator_criterion = generator_criterion.to(device)
    else:
        netG.cpu()
        netD.cpu()
        generator_criterion.cpu()

    # Adam - 自适应学习率+适用非凸优化
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    if opt.amp:
        from apex import amp
        netG, optimizerG = amp.initialize(netG, optimizerG, opt_level=opt.amp_level, loss_scale=opt.loss_scale_g)
        netD, optimizerD = amp.initialize(netD, optimizerD, opt_level=opt.amp_level, loss_scale=opt.loss_scale_d)

    # 结果集 : loss score psnr（峰值信噪比） ssim（结构相似性）
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'train_fps': []}
    best_results = 0

    cudnn.benchmark = True
    for epoch in range(1, NUM_EPOCHS + 1):
        avt.t_start('epoch')
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'train_fps': 0}
        # 进入train模式
        netG.train()
        netD.train()
        # fps 统计方法
        step = 0
        fps_number = 0
        fps_count_start = False
        fps_start_time = 0
        avt.t_start('training')
        for data, target in train_loader:
            step += 1
            fps_start_time = time.time()
            if step == 11:
                fps_count_start = True
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            if opt.save_prof and step == 10 and epoch == 1:
                with torch.autograd.profiler.profile(**prof_kwargs) as prof:
                    ############################
                    # (1) Update D network: maximize D(x)-1-D(G(z))
                    ###########################
                    real_img = Variable(target)
                    if opt.use_npu or opt.use_gpu:
                        real_img = real_img.to(device)
                    else:
                        real_img.cpu()
                    z = Variable(data)
                    if opt.use_npu or opt.use_gpu:
                        z = z.to(device)
                    else:
                        z.cpu()
                    fake_img = netG(z)

                    netD.zero_grad()
                    real_out = netD(real_img).mean()
                    fake_out = netD(fake_img).mean()
                    d_loss = 1 - real_out + fake_out
                    if opt.amp:
                        # with amp.scale_loss(d_loss, optimizerD, loss_id=0) as d_loss:
                        #     d_loss.backward(retain_graph=True)
                        with amp.scale_loss(d_loss, optimizerD) as scaled_d_loss:
                            scaled_d_loss.backward(retain_graph=True)
                    else:
                        d_loss.backward(retain_graph=True)
                    # 进行参数优化
                    optimizerD.step()

                    ############################
                    # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                    ###########################
                    netG.zero_grad()
                    ## The two lines below are added to prevent runetime error in Google Colab ##
                    fake_img = netG(z)
                    fake_out = netD(fake_img).mean()
                    g_loss = generator_criterion(fake_out.float(), fake_img.float(), real_img.float())
                    if opt.amp:
                        with amp.scale_loss(g_loss, optimizerG) as scaled_g_loss:
                            scaled_g_loss.backward()
                    else:
                        g_loss.backward()
                    #fake_img = netG(z)
                    #fake_out = netD(fake_img).mean()
                    optimizerG.step()
                # 保存 prof 文件
                prof.export_chrome_trace(f'{config.get_root_path()}srgan_prof_{device}_{step}.prof')
            else:
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                if opt.use_npu or opt.use_gpu:
                    real_img = real_img.to(device)
                else:
                    real_img.cpu()
                z = Variable(data)
                if opt.use_npu or opt.use_gpu:
                    z = z.to(device)
                else:
                    z.cpu()
                fake_img = netG(z)

                netD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()
                d_loss = 1 - real_out + fake_out

                if opt.amp:
                    with amp.scale_loss(d_loss, optimizerD) as scaled_d_loss:
                        scaled_d_loss.backward(retain_graph=True)
                else:
                    d_loss.backward(retain_graph=True)
                # 进行参数优化
                optimizerD.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                netG.zero_grad()
                ## The two lines below are added to prevent runetime error in Google Colab ##
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()
                g_loss = generator_criterion(fake_out.float(), fake_img.float(), real_img.float())
                if opt.amp:
                    with amp.scale_loss(g_loss, optimizerG) as scaled_g_loss:
                        scaled_g_loss.backward()
                else:
                    g_loss.backward()
                #fake_img = netG(z)
                #fake_out = netD(fake_img).mean()
                optimizerG.step()
            if fps_count_start:
                fps = batch_size / (time.time() - fps_start_time)
                fps = round(fps, 2)
                fps_number += 1
                running_results['train_fps'] += fps
            else:
                fps = 0

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size


            Loss_D = running_results['d_loss'] / running_results['batch_sizes']
            Loss_G = running_results['g_loss'] / running_results['batch_sizes']
            score_D = running_results['d_score'] / running_results['batch_sizes']
            score_G = running_results['g_score'] / running_results['batch_sizes']
            print(f'[{epoch}/{NUM_EPOCHS}] step:{step} Loss_D: {Loss_D:.4f} Loss_G: {Loss_G:.4f} '
                  f'D(x): {score_D:.4f} D(G(z)): {score_G:.4f} Fps: {fps:.4f}')

            avt.step_update()
        avt.print_time('training')
        running_results['train_fps'] = running_results['train_fps'] / fps_number
        opt.save_prof = False  # 关闭保存prof

        if not opt.performance:
            avt.t_start('val')
            netG.eval()
            with torch.no_grad():
                # val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                print_step = 0
                for val_lr, val_hr_restore, val_hr in val_loader:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if opt.use_npu or opt.use_gpu:  # 可以考虑使用cpu进行验证
                        lr = lr.to(device)
                        hr = hr.to(device)
                    sr = netG(lr)  # 使用网络生成的图像尺寸和原图尺寸不一致

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10(
                        (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    if print_step % 10 == 0 or print_step == 0:
                        psnr = valing_results['psnr']
                        ssim = valing_results['ssim']
                        print(f'[converting LR images to SR images] PSNR: {psnr:4f} dB SSIM: {ssim:4f}')
                    print_step += 1
                    if opt.save_val_img:
                        val_images.append(display_transform()(val_hr_restore.squeeze(0)))
                        val_images.append(display_transform()(hr.data.cpu().squeeze(0)))
                        val_images.append(display_transform()(sr.data.cpu().squeeze(0)))
                if opt.save_val_img:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // 15)
                    print('[saving training results]:\n')
                    index = 1
                    for image in val_images:
                        image = utils.make_grid(image, nrow=3, padding=5)
                        utils.save_image(image, val_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                        index += 1
            avt.print_time('val')
        else:
            print('Skip the verification process!')

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['train_fps'].append(running_results['train_fps'])
        if not opt.performance:
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])
        else:
            results['psnr'].append(0)
            results['ssim'].append(0)


        if opt.only_keep_best and not opt.performance:
            current_results = valing_results['psnr']/10 + valing_results['ssim']
            if current_results > best_results:
                print(f'current best validation results -> psnr: {valing_results["psnr"]}, '
                      f'ssim: {valing_results["ssim"]}')
                # save model parameters
                torch.save(netG.state_dict(), out_path + '/netG_best.pth')
                torch.save(netD.state_dict(), out_path + '/netD_best.pth')
                best_results = current_results
        else:
            if epoch == 1 or epoch % 5 == 0:
                # save model parameters
                torch.save(netG.state_dict(), out_path + '/netG_epoch_%d.pth' % (epoch))
                torch.save(netD.state_dict(), out_path + '/netD_epoch_%d.pth' % (epoch))

        with open(config.get_root_path() + 'epoch_log_1p.txt', 'w', encoding='utf-8') as f:
            title = 'epoch \t d_loss \t g_loss \t d_score \t g_score \t psnr \t ssim \t train_fps \n'
            f.write(title)
            print('saving to file')
            for i in range(0, epoch):
                str = f"{i + 1} \t {results['d_loss'][i]:.4f} \t {results['g_loss'][i]:.4f} \t " \
                      f"{results['d_score'][i]:.4f} \t {results['g_score'][i]:.4f} \t " \
                      f"{results['psnr'][i]:.4f} \t " \
                      f"{results['ssim'][i]:.4f} \t {results['train_fps'][i]:.4f} \n"
                f.write(str)
            print('write results successfully!')
        avt.print_time('epoch')
    avt.print_time('end')

