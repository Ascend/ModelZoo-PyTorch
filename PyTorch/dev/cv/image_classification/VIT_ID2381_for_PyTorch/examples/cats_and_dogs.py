#!/usr/bin/env python
# coding: utf-8
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

# # Visual Transformer with Linformer
# 
# Training Visual Transformer on *Dogs vs Cats Data*
# 
# * Dogs vs. Cats Redux: Kernels Edition - https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
# * Base Code - https://www.kaggle.com/reukki/pytorch-cnn-tutorial-with-cats-and-dogs/
# * Effecient Attention Implementation - https://github.com/lucidrains/vit-pytorch#efficient-attention

# In[1]:


#get_ipython().system('pip -q install vit_pytorch linformer')


# ## Import Libraries

# In[2]:


#from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import time
from vit_pytorch.efficient import ViT
import torch.npu
import os
import apex
try:
    from apex import amp
except:
    amp = None

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


# In[3]:


print(f"Torch: {torch.__version__}")


# In[4]:


# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42


# In[5]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# In[6]:


device = 'npu'


# ## Load Data

# In[7]:


os.makedirs('data', exist_ok=True)


# In[8]:


train_dir = './data/train'
test_dir = './data/test'


# In[9]:


with zipfile.ZipFile('./data/train.zip') as train_zip:
    train_zip.extractall('./data')
    
with zipfile.ZipFile('./data/test.zip') as test_zip:
    test_zip.extractall('./data')


# In[10]:


train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))


# In[11]:


print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")


# In[12]:


labels = [path.split('/')[-1].split('.')[0] for path in train_list]


# ## Random Plots

# In[13]:


random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)


# ## Split

# In[14]:


train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)


# In[15]:


print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")


# ## Image Augumentation

# In[16]:


train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


# ## Load Datasets

# In[17]:


class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


# In[18]:


train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)


# In[19]:


train_loader = DataLoader(dataset = train_data, batch_size=batch_size,num_workers=64, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size,num_workers=64,shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size,num_workers=64, shuffle=True)


# In[20]:


print(len(train_data), len(train_loader))


# In[21]:


print(len(valid_data), len(valid_loader))


# ## Effecient Attention

# ### Linformer

# In[22]:


efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)


# ### Visual Transformer

# In[23]:


model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(f'npu:{NPU_CALCULATE_DEVICE}')


# ### Training

# In[24]:


# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=lr)
model.npu()
model, optimizer = amp.initialize(model, optimizer, opt_level = 'O2', loss_scale = 128.0, combine_grad=True)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


# In[25]:


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    step = 0
    for data, label in tqdm(train_loader):
        if step > 10:
          pass
        start_time = time.time()
        data = data.to(f'npu:{NPU_CALCULATE_DEVICE}')
        label = label.to(f'npu:{NPU_CALCULATE_DEVICE}')

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        #loss.backward()
        with amp.scale_loss(loss,optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        step += 1
        step_time = time.time() - start_time
        FPS = batch_size / step_time
        print(
        f"Epoch : {epoch+1} - step : {step} - Loss : {loss:.4f} - time/step(s):{step_time:.4f} - FPS:{FPS:.3f}\n"
    )        

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(f'npu:{NPU_CALCULATE_DEVICE}')
            label = label.to(f'npu:{NPU_CALCULATE_DEVICE}')

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )


# In[ ]:




