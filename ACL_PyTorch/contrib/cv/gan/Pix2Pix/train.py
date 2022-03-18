# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
"""!!!!!!!!!!!!!!!npu修改的地方!!!!!!!!!!!!!!!!!!1"""
# import torch
import torch.npu
import os
LOCAL_RANK = int(os.getenv('LOCAL_RANK', 0))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', 0))
"""!!!!!!!!!!!!!!!性能优化!!!!!!!!!!!!!!!!!!1"""
WORLD_SIZE = int(os.getenv('RANK_SIZE', 1))

# python train.py --dataroot ./datasets/facades --name facades_pix2pix_mixed_precision --model pix2pix --direction BtoA
# python -m visdom.server
# UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.  pytorch 1.5.0不会报错
# gpu单卡
# python -m torch.distributed.launch --nproc_per_node 1 train.py --dataroot ./datasets/facades --name facades_pix2pix_mixed_precision_1p_batchsize1 --model pix2pix --direction BtoA --gpu_ids 0 --batch_size 1 --norm instance
# gpu多卡
# python -m torch.distributed.launch --nproc_per_node 2 train.py --dataroot ./datasets/facades --name facades_pix2pix_mixed_precision --model pix2pix --direction BtoA --gpu_ids 0,1 --batch_size 32 --norm instance --display_freq 384

# npu命令
# python -m torch.distributed.launch --nproc_per_node 1 train.py --dataroot ./datasets/facades --name facades_pix2pix_npu1p_batchsize1_lr0002 --model pix2pix --direction BtoA --gpu_ids 0 --batch_size 1  --norm instance 
# python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_npu8p_batchsize1_lr0016 --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance --lr 0.0016 --display_freq 384
# python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_npu8p_batchsize1_lr0002 --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance 

# 8p batchsize=1 lr=0002 epoch=800   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep800 --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance --lr 0.0002 --n_epochs_decay 700 --display_freq 50
# 2p batchsize=1 lr=0002 epoch=200   python -m torch.distributed.launch --nproc_per_node 2 train.py --dataroot ./datasets/facades --name facades_pix2pix_2p_bs1_lr0002_ep200 --model pix2pix --direction BtoA --gpu_ids 0,1 --batch_size 1 --norm instance --lr 0.0002  --display_freq 200
# 8p batchsize=1 lr=0002 epoch=1200   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep1200 --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance --lr 0.0002 --n_epochs_decay 1100 --display_freq 50
# 8p batchsize=1 lr=0002 epoch=1200   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep1200_warmup --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance --lr 0.0002 --n_epochs_decay 1100 --display_freq 50 --lr_policy warm_up
# 8p batchsize=1 lr=0002 epoch=200   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep200 --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance  --display_freq 50 --print_freq 1 --display_id -1
# 8p batchsize=1 lr=0002 epoch=200   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep200_ --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance  --display_freq 50 --print_freq 1 --display_id -1
# 8p batchsize=1 lr=0002 epoch=200   python -m torch.distributed.launch --nproc_per_node 8 train.py --dataroot ./datasets/facades --name facades_pix2pix_8p_bs1_lr0002_ep200_ --model pix2pix --direction BtoA --gpu_ids 0,1,2,3,4,5,6,7 --batch_size 1 --norm instance  --display_freq 50 --print_freq 1 --display_id -1

if __name__ == '__main__':
    """!!!!!!!!!!!!!!!npu8p修改的地方!!!!!!!!!!!!!!!!!!1"""
    torch.manual_seed(4)
    """!!!!!!!!!!!!!!!npu8p修改的地方!!!!!!!!!!!!!!!!!!1"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29139'
    opt = TrainOptions().parse()   # get training options
    """!!!!!!!!!!!!!!!修改的地方!!!!!!!!!!!!!!!!!!1"""
    """添加的内容,用0，1两张卡,world_size=torch.cuda.device_count()
              rank=0表示主机 rank是标识主机和从机的, 这里就一个主机, 设置成0就行了.
              world_size是标识使用几个主机, 这里就一个主机, 设置成1就行了, 设置多了代码不允许.
    """
    """!!!!!!!!!!!!!!!npu修改的地方!!!!!!!!!!!!!!!!!!1"""
    """!!!!!!!!!!!!!!!npu8p修改的地方!!!!!!!!!!!!!!!!!!1"""
    # torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(badckend="hccl")
    """!!!!!!!!!!!!!!!性能优化!!!!!!!!!!!!!!!!!!1"""
    #dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='29139 ') # master_port取两万开头的
    #torch.distributed.init_process_group(backend="hccl", world_size=WORLD_SIZE,rank=RANK,init_method=dist_init_method)
    torch.distributed.init_process_group(backend="hccl", world_size=WORLD_SIZE,rank=RANK)
    print(f"[init] == local rank: {LOCAL_RANK}, global rank: {RANK} , world size: {WORLD_SIZE}")

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        """!!!!!!!!!FPS!!!!!!!!!!!"""
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        """!!!!!!!!!!!!!!!修改的地方!!!!!!!!!!!!!!!!!!1"""
        dataset.train_sampler.set_epoch(epoch)

       
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            """!!!!!!!FPS!!!!!!!!"""
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()

                """!!!!!!!FPS!!!!!!!!"""
                t_comp_no_t_data = (time.time()-iter_start_time)
                FPS = int(opt.batch_size / t_comp_no_t_data)


                t_comp = (time.time() - iter_start_time) / opt.batch_size
                """!!!!!!!FPS!!!!!!!!"""
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data,FPS)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

           

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
