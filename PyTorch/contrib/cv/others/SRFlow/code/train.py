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

import argparse
import logging
import os
import random
import time
from os.path import basename

import cv2
import math
import torch
if torch.__version__>="1.8":
    import torch_npu
print(torch.__version__)
import torch.distributed as dist

import options.options as option
from data import create_dataloader, create_dataset
from models import create_model
from utils import util
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths


def getEnv(name): import os; return True if name in os.environ.keys() else False


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--performance',action='store_true')
    parser.add_argument('--finetune', type=str, default='none')

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['local_rank'] = args.local_rank

    opt['dist'] = False
    rank = -1
    # the 'WORLD_SIZE' environment variable will also be set automatically.（torch.distributed.launch）
    if 'WORLD_SIZE' in os.environ:
        opt['dist'] = True
        opt['num_dev'] = int(os.environ['WORLD_SIZE'])
    else:
        opt['num_dev'] = torch.npu.device_count()
    opt['use_amp'] = True
    if opt['dist']:
        torch.npu.set_device("npu:{}".format(args.local_rank))
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # You can use the current real ip or '127.0.0.1'
        os.environ['MASTER_PORT'] = '29688'  # Feel free to use a port
        torch.distributed.init_process_group(backend='hccl',world_size=opt['num_dev'],rank=args.local_rank)
        rank = args.local_rank

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)
        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        else:
            device_id = torch.npu.current_device()
            resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage)
            option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            print('version', version)
            from torch.utils.tensorboard import SummaryWriter
            print('basename(args.opt)', basename(args.opt))
            conf_name = basename(args.opt).replace(".yml", "")
            print('conf_name', conf_name)
            exp_dir = opt['path']['experiments_root']
            print('exp_dir', exp_dir)
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
            print('log_dir_train', log_dir_train)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    print('# convert to NoneDict, which returns None for missing keys')
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            print('Dataset created')
            print('### train_set', len(train_set))
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            # total_iters = int(opt['train']['niter'])
            # total_epochs = int(math.ceil(total_iters / train_size))
            total_epochs = opt['train']['total_epochs']
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            # calculate total_iters according to total_epochs and adjust the learning rate
            opt['train']['niter'] = len(train_loader) * total_epochs
            total_iters = opt['train']['niter']

            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # relative learning rate
    if 'train' in opt:
        niter = opt['train']['niter']
        if 'T_period_rel' in opt['train']:
            opt['train']['T_period'] = [int(x * niter) for x in opt['train']['T_period_rel']]
        if 'restarts_rel' in opt['train']:
            opt['train']['restarts'] = [int(x * niter) for x in opt['train']['restarts_rel']]
        if 'lr_steps_rel' in opt['train']:
            opt['train']['lr_steps'] = [int(x * niter) for x in opt['train']['lr_steps_rel']]
        if 'lr_steps_inverse_rel' in opt['train']:
            opt['train']['lr_steps_inverse'] = [int(x * niter) for x in opt['train']['lr_steps_inverse_rel']]
        print(opt['train'])
    # create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        epoch_time = resume_state['epoch_time']
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        epoch_time = 0
    # finetune
    if args.finetune.endswith('.pth'):
        model.load_network(load_path=args.finetune, network=model.netG)
    len_train_loader = len(train_loader)
    epoch_starttime = 0
    # training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()
    for epoch in range(start_epoch, total_epochs):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        timerData.tick()
        for step, train_data in enumerate(train_loader):
            timerData.tock()
            current_step += 1
            model.feed_data(train_data)
            nll = None
            try:
                nll = model.optimize_parameters(current_step)
            except RuntimeError as e:
                print("Skipping ERROR caught in nll = model.optimize_parameters(current_step): ")
                print(e)

            if nll is None:
                nll = 0
            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # log
            def eta(t_iter):
                return (t_iter * (opt['train']['niter'] - current_step)) / 3600

            # Do not count the time spent in the first 5 steps of each epoch, start timing after running 0-4
            if step >= 4 and epoch_starttime == 0:
                epoch_starttime = time.time()
            if rank <= 0 and (step % opt['logger']['print_freq'] == 0
                              or current_step - (resume_state['iter'] if resume_state else 0) < 25):
                avg_time = timer.get_average_and_reset()
                avg_data_time = timerData.get_average_and_reset()
                message = '<epoch:{:3d}/{:3d}, iter:{:8,d}/{:d}, lr:{:.3e}, t:{:.2e}, td:{:.2e}, eta:{:.2e}, nll:{:.3e}> '.format(
                    epoch, total_epochs, step, len(train_loader), model.get_current_learning_rate(), avg_time,
                    avg_data_time,
                    eta(avg_time), nll)
                print(message)
            timer.tick()
            # Reduce number of logs
            if current_step % 5 == 0 and opt['use_tb_logger']:
                tb_logger_train.add_scalar('loss/nll', nll, current_step)
                tb_logger_train.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                tb_logger_train.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                for k, v in model.get_current_log().items():
                    tb_logger_train.add_scalar(k, v, current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                nlls = []
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)

                    nll = model.test()
                    if nll is None:
                        nll = 0
                    nlls.append(nll)

                    visuals = model.get_current_visuals()

                    sr_img = None
                    # Save SR images for reference
                    if hasattr(model, 'heats'):
                        for heat in model.heats:
                            for i in range(model.n_sample):
                                sr_img = util.tensor2img(visuals['SR', heat, i])  # uint8
                                save_img_path = os.path.join(img_dir,
                                                             '{:s}_{:09d}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                                    current_step,
                                                                                                    int(heat * 100), i))
                                util.save_img(sr_img, save_img_path)
                    else:
                        sr_img = util.tensor2img(visuals['SR'])  # uint8
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path)
                    assert sr_img is not None

                    # Save LQ images for reference
                    save_img_path_lq = os.path.join(img_dir,
                                                    '{:s}_LQ.png'.format(img_name))
                    if not os.path.isfile(save_img_path_lq):
                        lq_img = util.tensor2img(visuals['LQ'])  # uint8
                        util.save_img(
                            cv2.resize(lq_img, dsize=None, fx=opt['scale'], fy=opt['scale'],
                                       interpolation=cv2.INTER_NEAREST),
                            save_img_path_lq)

                    # Save GT images for reference
                    gt_img = util.tensor2img(visuals['GT'])  # uint8
                    save_img_path_gt = os.path.join(img_dir,
                                                    '{:s}_GT.png'.format(img_name))
                    if not os.path.isfile(save_img_path_gt):
                        util.save_img(gt_img, save_img_path_gt)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                avg_psnr = avg_psnr / idx
                avg_nll = sum(nlls) / len(nlls)

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                if opt['use_tb_logger']:
                    # tensorboard logger
                    tb_logger_valid.add_scalar('loss/psnr', avg_psnr, current_step)
                    tb_logger_valid.add_scalar('loss/nll', avg_nll, current_step)

                    tb_logger_train.flush()
                    tb_logger_valid.flush()

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step,
                                              time.time() - epoch_starttime + epoch_time)

            timerData.tick()
            performance_step = 100
            if args.performance and step==performance_step:
                message = '<test performance of 100 iters>'
                print(message)
                break

        if rank <= 0:
            time_all = time.time() - epoch_starttime + epoch_time
            if args.performance:
                time_avg = time_all / (performance_step - 5)
            else:
                time_avg = time_all / (len_train_loader - 5)
            fps = opt['datasets']['train']['batch_size'] * opt['num_dev'] / time_avg
            message = '<epoch:{:3d}, fps:{:f}> '.format(epoch, fps)
            print(message)
        epoch_starttime = 0
        epoch_time = 0
        if args.performance:
            break
    with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
        f.write("TRAIN_DONE")

    if rank <= 0 and not args.performance:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
