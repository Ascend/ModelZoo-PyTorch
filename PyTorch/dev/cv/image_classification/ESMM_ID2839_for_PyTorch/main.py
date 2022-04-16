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
# @Time    : 2021-04-19 17:25
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  script description


from utils import data_preparation, TrainDataSet
from torch.utils.data import DataLoader
from model_train import train_model
from esmm import ESMM
from mmoe import MMOE
import torch
import torch.nn as nn
import torch.npu
import os
from apex import amp
import apex
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


def main():
    train_data, test_data, user_feature_dict, item_feature_dict = data_preparation()
    train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
    # val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
    test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
    train_dataset = TrainDataSet(train_dataset)
    # val_dataset = TrainDataSet(val_dataset)
    test_dataset = TrainDataSet(test_dataset)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # pytorch浼樺寲鍙傛暟
    learn_rate = 0.01
    bce_loss = nn.BCEWithLogitsLoss()
    early_stop = 3
    epoch = 10
    
    # train model
    # esmm Epoch 17 val loss is 1.164, income auc is 0.875 and marry auc is 0.953
    esmm = ESMM(user_feature_dict, item_feature_dict, emb_dim=64).to(f'npu:{NPU_CALCULATE_DEVICE}')
    optimizer = apex.optimizers.NpuFusedAdam(esmm.parameters(), lr=learn_rate)
    esmm, optimizer = amp.initialize(esmm, optimizer, opt_level='O2', loss_scale=128.0, combine_grad=True)
    train_model(esmm, train_dataloader, test_dataloader, epoch, bce_loss, optimizer, 'model/model_esmm_{}', early_stop)
    
    # mmoe
    #mmoe = MMOE(user_feature_dict, item_feature_dict, emb_dim=64)
    #optimizer = torch.optim.Adam(mmoe.parameters(), lr=learn_rate)
    #train_model(mmoe, train_dataloader, test_dataloader, epoch, bce_loss, optimizer, 'model/model_mmoe_{}', early_stop)
    

if __name__ == "__main__":
    main()
