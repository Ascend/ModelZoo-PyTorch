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
import os
import time
import torch
from dataloader import create_dataset
from parse import parse_args
from util.visualizer_adapt import Visualizer
import torch.multiprocessing as mp
from models.cycle_gan_model_adapt import CycleGANModel as create_model
from torch import distributed as dist


def main(opt):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23112'
    if opt.distributed >= 1:
        ngpus_per_node = len(opt.process_device_map)
        opt.ngpus_per_node = ngpus_per_node
        if (ngpus_per_node == 1):
            ngpus_per_node = 0
    opt.total_iters = 0
    if opt.multiprocessing_distributed >= 1:
        opt.world_size = ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(ngpus_per_node, opt, opt)


def main_worker(gpu, ngpus_per_node, args):
    opt = args
    print([args.process_device_map, gpu])
    opt.gpu = args.process_device_map[gpu]
    if (opt.distributed >= 1):
        opt.rank = gpu
        if opt.multiprocessing_distributed >= 1:
            opt.rank = gpu
        if (opt.npu < 1):
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=opt.world_size,
                                                 rank=opt.rank)
        elif (opt.npu >= 1):
            torch.npu.set_device(gpu)
            torch.distributed.init_process_group(backend='hccl', world_size=opt.world_size, rank=opt.rank)
    dataset, train_sampler = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    opt.isTrain = True
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    for epoch in range(opt.num_epoch_start, opt.num_epoch):
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        if (opt.ngpus_per_node > 1):
            train_sampler.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            opt.total_iters += (opt.batch_size * opt.ngpus_per_node)
            if (opt.prof >= 1 and i > 10):
                if (opt.npu == False):
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        model.set_input(data)
                        model.optimize_parameters()
                    print(prof.key_averages().table())
                    prof.export_chrome_trace(opt.prof_file)  # "output.prof"
                    opt.prof = False
                else:
                    with torch.autograd.profiler.profile(use_npu=True) as prof:
                        model.set_input(data)
                        model.optimize_parameters()
                    print(prof.key_averages().table())
                    prof.export_chrome_trace(opt.prof_file)  #
                    opt.prof = False
            else:
                model.set_input(data)
                model.optimize_parameters()
            if opt.total_iters % opt.save_latest_freq == 0:  # print training losses and save logging information to the disk
                model.save_networks(epoch)
                # model.save_networks(epoch)
            if opt.total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                fps = opt.batch_size * opt.ngpus_per_node / t_comp
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, fps, losses, t_comp)
                # print_current_losses(opt, epoch, fps, losses, t_comp)
                save_result = opt.total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
        model.update_learning_rate()  # Update learning rates

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, opt.total_iters))
            model.save_networks('latest_pu' + str(opt.gpu))
            model.save_networks(str(epoch) + '_pu' + str(opt.gpu))
        dist.barrier()


if __name__ == '__main__':
    paser = parse_args(True, False)
    opt = paser.initialize()
    paser.printParser()
    main(opt)
