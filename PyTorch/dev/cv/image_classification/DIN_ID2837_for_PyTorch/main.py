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
import torch
import torch.nn as nn
import os
import sys
import pickle as pk
import numpy as np
import random

from sklearn.metrics import roc_auc_score
import time
import torch.npu
import os
import apex
from apex import amp
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

workspace_dir = '.'
try:
    from google.colab import drive
    drive.mount( '/content/drive/' )

    workspace_dir = os.path.join( '.' , 'drive', 'My Drive', 'DIN-pytorch')
    sys.path.append( workspace_dir)
    #! rm -rf data
    #! tar zxf "{workspace_dir/npu/traindata/ID2837_CarPeting_Pytorch_DIN.tar.gz" -C ./
    #! tar zxf "{workspace_dir}/loader.tar.gz" -C ./
    #! ls -al data   
except ImportError:
    pass

from model import DIN, DIEN, DynamicGRU
from DataLoader import MyDataSet


#Model hyper parameter
MAX_LEN = 100
EMBEDDING_DIM = 32
# HIDDEN_SIZE_ATTENTION = [80, 40]
# HIDDEN_SIZE_FC = [200, 80]
# ACTIVATION_LAYER = 'LeakyReLU' # lr = 0.01


# Adam
LR = 1e-3
BETA1 = 0.5
BETA2 = 0.99

# Train
BATCH_SIZE = 128
EPOCH_TIME = 20
TEST_ITER = 1000

RANDOM_SEED = 19940808

USE_CUDA = True

train_file = os.path.join( '/npu/traindata/ID2837_CarPeting_Pytorch_DIN', "local_train_splitByUser")
test_file  = os.path.join( '/npu/traindata/ID2837_CarPeting_Pytorch_DIN', "local_test_splitByUser")
uid_voc    = os.path.join( '/npu/traindata/ID2837_CarPeting_Pytorch_DIN', "uid_voc.pkl")
mid_voc    = os.path.join( '/npu/traindata/ID2837_CarPeting_Pytorch_DIN', "mid_voc.pkl")
cat_voc    = os.path.join( '/npu/traindata/ID2837_CarPeting_Pytorch_DIN', "cat_voc.pkl")

if USE_CUDA and torch.npu.is_available():
    print( "Cuda is avialable" )
    device = torch.device(f'npu:{NPU_CALCULATE_DEVICE}')
    dtype = torch.npu.FloatTensor
else:
    device = torch.device( f'npu:{NPU_CALCULATE_DEVICE}')
    dtype = torch.FloatTensor

# Stable the random seed
def same_seeds(seed = RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  
    random.seed(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Initilize  parameters
def weights_init( m):
    try:
        classname = m.__class__.__name__
        if classname.find( 'BatchNorm') != -1:
            nn.init.normal_( m.weight.data, 1.0, 0.02)
            nn.init.constant_( m.bias.data, 0)
        elif classname.find( 'Linear') != -1:
            nn.init.normal_( m.weight.data, 0.0, 0.02)
        elif classname.find( 'Embedding') != -1:
            m.weight.data.uniform_(-1, 1)
    except AttributeError:
        print( "AttributeError:", classname)
    


def eval_output( scores, target, loss_function = torch.nn.functional.binary_cross_entropy_with_logits):
    loss = loss_function( scores, target)

    y_pred = scores.sigmoid().round()

    accuracy = ( y_pred == target).type( dtype).mean()

    auc = roc_auc_score( target.cpu().detach(), scores.cpu().detach() )
    return loss, accuracy, auc

# The dict mapping description(string) to type index(int) 
# A more graceful api https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder not used in this project

user_map = pk.load( open( uid_voc, 'rb')); n_uid = len( user_map)
material_map = pk.load( open( mid_voc, 'rb')); n_mid = len( material_map)
category_map = pk.load( open( cat_voc, 'rb')); n_cat = len( category_map)

same_seeds( RANDOM_SEED)

dataset_train = MyDataSet( train_file, user_map, material_map, category_map, max_length = MAX_LEN)
dataset_test = MyDataSet( test_file, user_map, material_map, category_map, max_length = MAX_LEN)

loader_train = torch.utils.data.DataLoader( dataset_train, batch_size = BATCH_SIZE, shuffle = True, pin_memory=True)
loader_test = torch.utils.data.DataLoader( dataset_test, batch_size = BATCH_SIZE, shuffle = False)

# with open( 'data/loader.pk', 'rb') as fin:
#     loader_train, loader_test = pk.load(fin) 

# Get model and initialize it
# model = DIEN(  n_uid, n_mid, n_cat, EMBEDDING_DIM).to( device)
model = DIN(  n_uid, n_mid, n_cat, EMBEDDING_DIM ).to( f'npu:{NPU_CALCULATE_DEVICE}')
model.apply( weights_init)

# Set loss function and optimizer
optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), LR, (BETA1, BETA2))
model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0, combine_grad=True)
model.train(); iter = 0
for epoch in range( EPOCH_TIME):
    for i, data in enumerate( loader_train):
        if i >= 1000:pass
        start_time = time.time()
        iter += 1

        # transform data to target device
   
        data = [ item.to( f'npu:{NPU_CALCULATE_DEVICE}', non_blocking=True) if item != None else None for item in data]
        target = data.pop(-1)     
        
        model.zero_grad()

        scores = model( data, neg_sample = False)
        
        loss, accuracy, auc = eval_output( scores, target)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step( )
        step_time = time.time() - start_time
        FPS = BATCH_SIZE / step_time
        print( "Epoch:{}, step:{}, loss:{:.4f}, Acc:{:.4f},Auc:{:.4f}, time/step(s):{:.4f},FPS:{:.3f}".format( epoch + 1, i + 1, loss.item(), accuracy.item(), auc.item(), step_time, FPS))

        if iter % TEST_ITER == 0:
            model.eval()
            with torch.no_grad():
                score_list = []; target_list = []
                for data in  loader_test:
                    data = [ item.to( f'npu:{NPU_CALCULATE_DEVICE}') if item != None else None for item in data]
                    
                    target = data.pop(-1)

                    scores = model( data, neg_sample = False)
                    score_list.append( scores)
                    target_list.append( target)
                scores = torch.cat( score_list, dim = -1)
                target = torch.cat( target_list, dim = -1)
                loss, accuracy, auc = eval_output( scores, target)
                print( "\tTest Set\tloss:%.5f\tacc:%.5f\tauc:%.5f"%( loss.item(), accuracy.item(), auc.item() ) )
            model.train()
