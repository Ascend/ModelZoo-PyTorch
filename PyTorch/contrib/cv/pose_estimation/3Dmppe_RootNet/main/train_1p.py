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
from config import cfg
import torch
import os.path as osp
from base import Trainer
import torch.backends.cudnn as cudnn
from timer import Timer
import apex
from apex import amp
import torch.multiprocessing as mp
import torch.npu
from tqdm import tqdm
import numpy as np
from base import Tester
from utils.vis import vis_keypoints
import cv2
import os
import torch.distributed as dist
CALCULATE_DEVICE = "npu"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet_type', default=50, type=int, 
                        help='choose which resnet to use : 50, 101, 152')
    parser.add_argument('--lr_dec_epoch', default=17, type=int, 
                        help='learning rate will drop from this epoch')
    parser.add_argument('--end_epoch', default=20, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_dec_factor', default=10, type=int, 
                        help='the multiple of the reduction in learning rate, '
                             'if 10, the learning rate drops from 0.001 to 0.0001')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='batch_size')
    parser.add_argument('--rank', default=0, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--continue', dest='continue_train', action='store_true',
                        help='continue train or not')
    parser.add_argument('--prof', dest='use_prof', action='store_true',
                        help='use prof or not')
    parser.add_argument('--distributed', action='store_true', 
                        help='use DDP or not')
    parser.add_argument("--world_size", default=1, type=int,  
                        help='number of nodes for distributed training')
    parser.add_argument("--num_thread", default=16, type=int,  
                        help='number of threads')
    parser.add_argument('--amp', default=False, action='store_true', 
                        help='use amp to train the model')
    parser.add_argument('--loss_scale', default=-1, type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt_level', default='O1', type=str,
                        help='opt_level using in amp')
    parser.add_argument('--npu_device', default='0', type=str,
                        help='specifies the id of the NPU to use')
    parser.add_argument('--step_print', default=20, type=int,
                        help='record information every 50 steps')
    parser.add_argument('--model_save_start', default=6, type=int,
                        help='Save the model from the 6th epoch')
    parser.add_argument('--model_save_interval', default=2, type=int,
                        help='the interval to save the model')
    parser.add_argument('--performance', default=False, action='store_true',
                        help='Use performance mode or not')
    parser.add_argument('--npu_device_test', default='0', type=str,
                        help='specifies the id of the npu to use')
    parser.add_argument('--data_path', default='data', type=str,
                        help='location of the dataset')
    args = parser.parse_args()

    args.npu_device = CALCULATE_DEVICE + ':' + args.npu_device
    args.npu_device_test = CALCULATE_DEVICE + ':' + args.npu_device_test
    npus_per_node = 1
    args.batch_size = int(args.batch_size / npus_per_node)
    args.world_size = npus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, args)) 
    
    return args


def main():

    # argument parse and create log
    parse_args()


def evalaute(epoch, args):

    tester = Tester(epoch)
    tester._make_batch_generator()
    tester._make_model(args.npu_device_test)

    preds = []
    with torch.no_grad():
        for itr, (input_img, cam_param) in enumerate(tqdm(tester.batch_generator)):
            input_img = input_img.to(args.npu_device_test, non_blocking=True)
            cam_param = cam_param.to(args.npu_device_test, non_blocking=True)
            coord_out = tester.model(input_img, cam_param).to(args.npu_device_test)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)

    # evaluate
    preds = np.concatenate(preds, axis=0)
    ap_root, result = tester._evaluate(preds, cfg.result_dir)
    return ap_root, result


def main_worker(rank, npus_per_node, args):

    trainer = Trainer()
    trainer.logger.info('Parameter Settings')
    trainer.logger.info(args)
    cfg.set_args_train1P(args.resnet_type, args.lr_dec_epoch, args.end_epoch, args.lr, args.lr_dec_factor,
                         args.batch_size, args.rank, args.continue_train, args.distributed, args.world_size,
                         args.num_thread, args.use_prof, npus_per_node, args.amp, args.loss_scale,
                         args.opt_level, args.npu_device, args.data_path)
    trainer.logger.info('ngpus_per_node:' + str(npus_per_node))
    cudnn.fastest = True
    cudnn.benchmark = True
    trainer.logger.info('Parameter Settings complete')
    trainer._make_model(rank)
    trainer._make_batch_generator()
    fps_timer = Timer()
    best_epoch = 0
    best_ap_root = 0

    if args.performance:
        trainer.logger.info('Performance mode')
    if cfg.use_prof:
        prof = torch.autograd.profiler.profile(use_npu=True)
    
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        if cfg.distributed:
            trainer.batch_generator.sampler.set_epoch(epoch)
        
        for itr, (input_img, k_value, root_img, root_vis, joints_have_depth) in enumerate(trainer.batch_generator):
                
            # 去掉前5个step的时间
            if itr > 4:
                fps_timer.tic()
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            if cfg.use_prof:
                with torch.autograd.profiler.profile(use_npu=True) as prof:
                    # forward
                    trainer.optimizer.zero_grad()
                    input_img, k_value = input_img.to(cfg.npu_device, non_blocking=True), k_value.to(cfg.npu_device,
                                                                                                     non_blocking=True)
                    root_img, root_vis = root_img.to(cfg.npu_device, non_blocking=True), root_vis.to(cfg.npu_device,
                                                                                                     non_blocking=True)
                    joints_have_depth = joints_have_depth.to(cfg.npu_device, non_blocking=True)
                    target = {'coord': root_img, 'vis': root_vis, 'have_depth': joints_have_depth}
                    loss_coord = trainer.model(input_img, k_value, target).to(cfg.npu_device) 
                    loss_coord = loss_coord.mean()

                    # backward
                    loss = loss_coord
                    if cfg.amp:
                        with apex.amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    trainer.optimizer.step()
                    
            else:
                # forward
                trainer.optimizer.zero_grad()
                input_img, k_value = input_img.to(cfg.npu_device, non_blocking=True), k_value.to(cfg.npu_device,
                                                                                                 non_blocking=True)
                root_img, root_vis = root_img.to(cfg.npu_device, non_blocking=True), root_vis.to(cfg.npu_device,
                                                                                                 non_blocking=True)
                joints_have_depth = joints_have_depth.to(cfg.npu_device, non_blocking=True)
                target = {'coord': root_img, 'vis': root_vis, 'have_depth': joints_have_depth}
                loss_coord = trainer.model(input_img, k_value, target).to(cfg.npu_device) 
                loss_coord = loss_coord.mean()
            
                # backward
                loss = loss_coord
                if cfg.amp:
                    with apex.amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                trainer.optimizer.step()
            
            trainer.gpu_timer.toc()
            if itr % args.step_print == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                                                       trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    '%s: %.4f' % ('loss_coord', loss_coord.detach()),
                    ]
                trainer.logger.info(' '.join(screen))

            if itr > 4:
                fps_timer.toc()
            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if args.performance and itr >= 500:
                break
        
        fps = ['Epoch: %d  FPS:%.2f' % (epoch, cfg.batch_size / fps_timer.average_time)]
        trainer.logger.info(' '.join(fps))

        if args.performance:
            break

        if cfg.use_prof:
            trainer.logger.info(prof.key_averages().table(sort_by="self_cpu_time_total"))
            file_path = osp.join(cfg.prof_dir, 'epoch{}_output.prof'.format(str(epoch)))
            prof.export_chrome_trace(file_path)

        # test and save model
        if epoch >= args.model_save_start and (epoch % args.model_save_interval == 0
                                               or epoch == cfg.end_epoch-1 or epoch == cfg.end_epoch-2):
            trainer.logger.info('save model...')
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'amp': amp.state_dict()
            }, epoch)
            ap_root, result = evalaute(epoch, args)
            trainer.logger.info(result)
            if ap_root > best_ap_root:
                best_ap_root = ap_root
                best_epoch = epoch

    if not args.performance:
        trainer.logger.info('best epoch: ' + str(best_epoch) + '   best_AP_root: ' + str(best_ap_root))
    else:
        trainer.logger.info('End of performance mode')


if __name__ == "__main__":
    main()
