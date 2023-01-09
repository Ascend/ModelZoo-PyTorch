#
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
#
import os
import glob
import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, set, batch_size
from util.eval_model import eval
import torch.npu
import os
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
	NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
	torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        n=0
        for i, data in enumerate(tqdm(trainloader)):
            n=n+1
            if n >= 100:
                pass
            start_time = time.time()
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data
            images, labels = images.npu(), labels.npu()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            step_time = time.time() - start_time
            fps = batch_size / step_time
            if i < 3 and epoch == 1:
                print("step_time: ", step_time)
        scheduler.step()

        # evaluation every epoch
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%, train loss: {:.3f}, fps: {:.3f}, step/time(s): {:.3f}'.format(
                    100. * raw_accuracy, 100. * local_accuracy, total_loss.item(), fps, step_time))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:

                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
        local_loss_avg\
            = eval(model, testloader, criterion, 'test', save_path, epoch)

        print(
            'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * local_accuracy))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
            writer.add_scalar('Test/raw_accuracy', raw_accuracy, epoch)
            writer.add_scalar('Test/local_accuracy', local_accuracy, epoch)
            writer.add_scalar('Test/raw_loss_avg', raw_loss_avg, epoch)
            writer.add_scalar('Test/local_loss_avg', local_loss_avg, epoch)
            writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
            writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

