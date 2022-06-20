# -*- coding: utf-8 -*-
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
# @Time    : 2021-04-19 17:10
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description : model train function

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import time
import torch.npu
import os
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

def train_model(model, train_loader, val_loader, epoch, loss_function, optimizer, path, early_stop):
    """
    pytorch model train function
    :param model: pytorch model
    :param train_loader: dataloader, train data loader
    :param val_loader: dataloader, val data loader
    :param epoch: int, number of iters
    :param loss_function: loss function of train model
    :param optimizer: pytorch optimizer
    :param path: save path
    :param early_stop: int, early stop number
    :return: None
    """
    # use GPU
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    model.to(f'npu:{NPU_CALCULATE_DEVICE}')
    
    # 澶氬皯姝ュ唴楠岃瘉闆嗙殑loss娌℃湁鍙樺皬灏辨彁鍓嶅仠姝
    patience, eval_loss = 0, 0
    
    # train
    for i in range(epoch):
        y_train_income_true = []
        y_train_income_predict = []
        y_train_marry_true = []
        y_train_marry_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2) in tqdm(enumerate(train_loader), total=len(train_loader)):
            start_time = time.time()
            x, y1, y2 = x.to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True), y1.to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True), y2.to(f'npu:{NPU_CALCULATE_DEVICE}',non_blocking=True)
            predict = model(x)
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            with amp.scale_loss(loss,optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            total_loss += float(loss)
            count += 1
            step_time = time.time() - start_time
            print("Epoch:{}, Step:{}, Loss:{:.4f}, time/step:{:4f}".format(i + 1,count,total_loss / count,step_time))
            y_train_income_true += list(y1.squeeze().cpu().numpy())
            y_train_marry_true += list(y2.squeeze().cpu().numpy())
            y_train_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())    
        torch.save(model.state_dict(), path.format(i + 1))
        income_auc = roc_auc_score(y_train_income_true, y_train_income_predict)
        marry_auc = roc_auc_score(y_train_marry_true, y_train_marry_predict)
        print("Epoch %d train loss is %.3f, income auc is %.3f and marry auc is %.3f" % (i + 1, total_loss / count,
                                                                                         income_auc, marry_auc))
        
        # 楠岃瘉
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_income_true = []
        y_val_marry_true = []
        y_val_income_predict = []
        y_val_marry_predict = []
        for idx, (x, y1, y2) in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y1, y2 = x.to(f'npu:{NPU_CALCULATE_DEVICE}'), y1.to(f'npu:{NPU_CALCULATE_DEVICE}'), y2.to(f'npu:{NPU_CALCULATE_DEVICE}')
            predict = model(x)
            y_val_income_true += list(y1.squeeze().cpu().numpy())
            y_val_marry_true += list(y2.squeeze().cpu().numpy())
            y_val_income_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_marry_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_eval_loss += float(loss)
            count_eval += 1
        income_auc = roc_auc_score(y_val_income_true, y_val_income_predict)
        marry_auc = roc_auc_score(y_val_marry_true, y_val_marry_predict)
        print("Epoch %d val loss is %.3f, income auc is %.3f and marry auc is %.3f" % (i + 1,
                                                                                       total_eval_loss / count_eval,
                                                                                       income_auc, marry_auc))
        
        # earl stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break
