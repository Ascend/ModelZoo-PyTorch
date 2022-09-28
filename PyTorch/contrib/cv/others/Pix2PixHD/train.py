# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import torch.nn as nn
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

opt = TrainOptions().parse()



def train(opt):

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    # 下面设置自动处理学习率以及最大epoch轮数的代码
    opt.world_size = int(os.environ['RANK_SIZE'])
    # 注意这里的batch_size是每一个卡上的batchsize，在我们的代码中每个卡上的batchsize为1
    # 下面的这句话也就是判断我们需要自动调整
    if opt.autoscale and (opt.batchSize * opt.world_size) != 1:
        factor = (opt.batchSize * opt.world_size) / 1
        if __name__ == '__main__':
            print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, opt.batchSize * opt.world_size))
        
        # 这个是调整学习的轮数
        # opt.niter //= factor
        # opt.niter = int(opt.niter)
        # opt.niter_decay //= factor
        # opt.niter_decay = int(opt.niter_decay)
        # opt.niter_fix_global //= factor
        # opt.niter_fix_global = int(opt.niter_fix_global)
        # 这个是调整学习率
        # opt.lr *= factor

    # 检查训练环境
    if opt.node_device == 'npu':
        if torch.npu.device_count() == 0:
            print('No NPUs detected. Exiting...')
            exit(-1)
    else:
        if torch.cuda.device_count() == 0:
            print('No GPUs detected. Exiting...')
            exit(-1)

    # 设置随机种子
    if opt.seed is not None:
        seed = opt.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        # 为cpu设置随机种子
        torch.manual_seed(seed)
        if opt.node_device == 'npu':
            # 为单个npu设置种子
            torch.npu.manual_seed(seed)
            # 为所有的npu设置种子
            torch.npu.manual_seed_all(seed)
            print("npu种子设置成功！")
        else:
            # 这里的gpu是类似的原理
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('Finish set seed, seed is :', seed)


    # 这个是判断我们的硬件设备是否存在
    if opt.node_device == 'npu':
        if torch.npu.is_available():
            print("npu environment is okay!, and current device count is", torch.npu.device_count())
    else:
        if torch.cuda.is_available():
            print("gpu environment is okay!, and current device count is", torch.cuda.device_count())
    # 首先我们对设备进行处理
    # 这个是确定某个进程的所用的卡编号
    opt.rank_id = int(os.environ['RANK_ID'])
    # 这个是所有进程的总共采用的卡数
    opt.world_size = int(os.environ['RANK_SIZE'])
    # 在这里具体实例化了某一台计算设备

    # 获取表示训练阶段的数字
    stage = int(os.environ['stage'])
    if opt.node_device == 'npu':
        opt.device = 'npu:' + str(opt.rank_id)
        torch.npu.set_device(opt.device)
    else:
        opt.device = 'cuda:' + str(opt.rank_id)
        torch.cuda.set_device(opt.device)
    
    # opt.is_master_node是一个bool量，用于指示哪个是主gpu或者npu设备
    opt.is_master_node = (opt.world_size == 1 or opt.rank_id == 0)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    dist.init_process_group(backend='hccl' if opt.node_device == 'npu' else 'nccl', world_size=opt.world_size, rank=opt.rank_id)

    # 下面正是开始模型的构建的过程

    # 首先是数据集的构建
    data_loader = CreateDataLoader(opt)
    # 这里拿到的是我们的标准pytorch中的dataloader
    # dataset = data_loader.load_data()
    dataset, train_sampler = data_loader.load_data()
    # 这里是返回一个epoch中有几个iteration，因为在本利中，我们设置其batchsize是1，因此这里的dataset_size就是2975
    dataset_size = len(data_loader)
    print('# training images = %d' % dataset_size)

    # 数据集在这里构建完成，数据的设备迁移工作源代码是放在了网络的前向传播部分
    # 因此上面获取数据的部分都是在cpu上进行运行的，我们网络的改进可以放在后半部分


    # 这里是模型的构建部分，在模型的构建部分中，既要涉及到模型的迁移设备工作，也要通过设计到数据的工作。
    # 改动之后创建出来的模型就都在对应设备上了
    # 这个创建出来的模型默认就是在gpu上的
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # 在之后我们的训练都是通过fp16来进行训练的
    if opt.fp16:    
        from apex import amp
        # model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1', loss_scale=128.0, combine_grad=True) 
        if stage == 1:
            model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1') 
        else:
            model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1', loss_scale=64.0)      
        # model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        # 这一句是实现多卡的核心代码
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.rank_id], broadcast_buffers=False)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
        

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    print("start_epoch = ", start_epoch)
    print("opt.niter + opt.niter_decay = ", opt.niter + opt.niter_decay)
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        # 下面这一句也是ddp所要求的代码
        train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        num_step = 0
        for i, data in enumerate(dataset, start=epoch_iter):
            # num_step += 1
            # if num_step == 2:
            #     exit(0)
            # 这里拿到的数据都是在cpu上
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################

            # print("程序运行到了这里!")
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                Variable(data['image']), Variable(data['feat']), device = opt.device, infer=save_fake)

            # torch.npu.synchronize()
            # sum per device losses
            losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
            loss_dict = dict(zip(model.module.loss_names, losses))
            

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: 
                    scaled_loss.backward()                
            else:
                loss_G.backward()          
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: 
                    scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step() 
            # torch.npu.synchronize()       

            ############## Display results and errors ##########
            ### print out errors
            if opt.is_master_node:
                if total_steps % opt.print_freq == print_delta:
                    errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
                    t = (time.time() - iter_start_time) / opt.print_freq
                    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    visualizer.plot_current_errors(errors, total_steps)
                    #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ### display output images
            if opt.is_master_node:
                if save_fake:
                    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                        ('synthesized_image', util.tensor2im(generated.data[0])),
                                        ('real_image', util.tensor2im(data['image'][0]))])
                    visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            # if opt.is_master_node:
            if total_steps % opt.save_latest_freq == save_delta:
                if opt.is_master_node:
                    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    model.module.save('latest')            
                    np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break
        
        # end of epoch 
        iter_end_time = time.time()
        if opt.is_master_node:
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        # if opt.is_master_node:
        if epoch % opt.save_epoch_freq == 0:
            # 这里增加一个判断逻辑，只有代码处于主节点时才保存代码
            if opt.is_master_node:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
                model.module.save('latest')
                model.module.save(epoch)
            
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()
            

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()

if __name__ == '__main__':
    train(opt)
