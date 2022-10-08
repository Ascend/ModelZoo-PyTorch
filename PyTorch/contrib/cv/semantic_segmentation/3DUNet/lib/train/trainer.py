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


from time import sleep
import numpy as np
import torch
from apex import amp
import time
#-------------------------
import torch.npu
#-------------------------

from numpy import *
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import WriterWithoutTensorboard
# from torch.cuda import set_device

import torch.distributed as dist

best_dsc = 0

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epochs = args.nEpochs

    if args.warm_up_epochs > 0 and epoch < args.warm_up_epochs:
        lr = args.lr * ((epoch + 1) / (args.warm_up_epochs + 1))
    else:
        alpha = 0
        cosine_decay = 0.5 * (
                1 + np.cos(np.pi * (epoch - args.warm_up_epochs) / (epochs - args.warm_up_epochs)))
        decayed = (1 - alpha) * cosine_decay + alpha
        lr = args.lr * decayed

    print("=> Epoch[%d] Setting lr: %.4f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        #self.writer = SummaryWriter(log_dir=args.log_dir + name_model, comment=name_model)
        self.writer = WriterWithoutTensorboard(args)
        
        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 0

    def training(self):
        best_dsc = 0
        for epoch in range(self.start_epoch, self.args.nEpochs):
            
            if self.args.world_size > 1:
                self.train_data_loader.sampler.set_epoch(epoch)
            adjust_learning_rate(self.optimizer, epoch, self.args)

            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            DSC = self.writer.data['val']['dsc'] / self.writer.data['val']['count']
            # is_best = DSC > best_dsc
            best_dsc = max(DSC, best_dsc)
            print("==>Best Dsc: ", best_dsc)


            if self.args.save is not None and ((epoch + 1) % self.save_frequency) and self.args.rank == 0:
            # if self.args.save is not None:
                if self.args.world_size > 1:
                    # self.model.module.save_checkpoint({self.args.save,
                    #                         epoch, val_loss, best_dsc, self.optimizer}, is_best)
                    self.model.module.save_checkpoint(self.args.save,
                                            epoch, val_loss,
                                            optimizer=self.optimizer)
                else:
                    self.model.save_checkpoint(self.args.save,
                            epoch, val_loss,
                            optimizer=self.optimizer)
            if self.args.world_size > 1:
                dist.barrier()
            self.writer.write_end_of_epoch(epoch)
            # Max = max(Max, MAX)
            # print("\nDsc MAX{} ".format(Max))
            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()
        #fps_final = 0

        end = time.time()
        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            if self.args.prof and batch_idx==5:
                with torch.autograd.profiler.profile(use_npu=True) as prof:  
                    self.optimizer.zero_grad()

                    input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                    #--------------------------------------------------------------------------------------
                    # input_tensor = input_tensor.to("npu:0", non_blocking=True).to(torch.float)
                    # target = target.to("npu:0", non_blocking=True).to(torch.float)
                    
                    
                    input_tensor = input_tensor.npu().to(torch.float)
                    target = target.npu().to(torch.float)
                    #--------------------------------------------------------------------------------------
                    input_tensor.requires_grad = True
                    output = self.model(input_tensor)
                    loss_dice, per_ch_score = self.criterion(output, target)

                    if self.args.amp:
                        with amp.scale_loss(loss_dice, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_dice.backward()
                    
                    self.optimizer.step()

                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                prof.export_chrome_trace("output.prof") # "output.prof"为输出文件地址
                exit()


            #torch.cuda.synchronize()
            #_------------------------------------------
            torch.npu.synchronize()
            #_------------------------------------------
            data_time = time.time() - end

            self.optimizer.zero_grad()
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)

            #--------------------------------------------------------------------------------------
            input_tensor = input_tensor.npu().to(torch.float)
            target = target.npu().to(torch.float)
            #--------------------------------------------------------------------------------------
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            loss_dice, per_ch_score = self.criterion(output, target)

            if self.args.amp:
                with amp.scale_loss(loss_dice, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_dice.backward()
            
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            per_ch_score  = mean(per_ch_score)
            # fps_final += self.args.world_size*self.args.batchSz/batch_time
            # fps_final = fps_final / 256
            #_------------------------------------------
            torch.npu.synchronize()
            #_------------------------------------------
            if self.args.rank == 0:
                batch_time = time.time()-end
                print("train {}/{} loss: {} score: {} batch_time: {} data_time: {} fps: {}".format(batch_idx, self.len_epoch, loss_dice, per_ch_score, batch_time, data_time, self.args.world_size*self.args.batchSz/batch_time))
            end = time.time()

            # exit()

            
            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')
        #print("[npu id:", "npu:0" "]",'* FPS@all {:.3f}'.format(fps_final))
        if self.args.rank == 0:
            self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
        
            input_tuple = [i.unsqueeze(1) for i in input_tuple[:-1]] + [input_tuple[-1]]

            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                
                #--------------------------------------------------------------------------------------
                input_tensor = input_tensor.npu().to(torch.float)
                target = target.npu().to(torch.float)
                #--------------------------------------------------------------------------------------
                input_tensor.requires_grad = False
                #import pdb; pdb.set_trace()
                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        if self.args.rank == 0:
            self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)



