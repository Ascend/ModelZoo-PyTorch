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

# ## Pytorch : 1. ANN & Simple CNN
# 
# > This Kernel is based on [Pytorch Tutorial for Deep Learning Lovers](https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers)
# 
# The project following the visualization is a deep learning model implementation project. Many people use the framework, but it is difficult to implement the paper. Through this project, I would like to develop the ability to implement papers.
# 
# - ANN
# - Simple CNN
# 

# In[1]:


import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import torch.npu
import os
import apex
try:
	from apex import amp
except ImportError:
	amp = None

NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


train = pd.read_csv("../datasets/Kannada-MNIST/train.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
target = train.label.values
train = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(X_train[10].reshape(28,28))
plt.axis("off")
plt.title(str(y_train[10]))
plt.savefig('graph.png')
plt.show()


# ## ANN (Artificial Neural Network)

# In[4]:


# Import Libraries
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


# In[5]:


# Create ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

# instantiate ANN
input_dim = 28*28

hidden_dim = 100 
output_dim = 10

# Create ANN
os.environ["CUDA_VISIBLE_DEVICES"]="0"
if torch.npu.is_available():
    print('cuda available')
    model = ANNModel(input_dim, hidden_dim, output_dim).npu()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.02
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=learning_rate)
model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O2',
                                      loss_scale='128.0',
                                      combine_grad=True)


# In[6]:


# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        start_time = time.time()
        train = Variable(images.view(-1, 28*28)).npu()
        labels = Variable(labels).npu()
        
        
        optimizer.zero_grad() # Clear gradients
        outputs = model(train) # Forward propagation
        loss = error(outputs, labels) # Calculate softmax and cross entropy loss
        # loss.backward() # Calculating gradients
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step() # Update parameters
        if i < 2:
            print("step_time = {:.4f}".format(time.time() - start_time), flush=True)       
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            
            # Predict test dataset
            for images, labels in test_loader:
                test = Variable(images.view(-1, 28*28)).npu()
                outputs = model(test) # Forward propagation
                predicted = torch.max(outputs.data, 1)[1].npu() # Get predictions from the maximum value
                total += len(labels) # Total number of labels
                correct += (predicted == labels.npu()).sum() # Total correct predictions
            
            accuracy = 100.0 * correct.item() / total
            
            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            step_time = time.time() - start_time
            FPS = batch_size / step_time
            print('Epoch:{}, Iteration: {}  total_loss: {:.4f}, time/step(s): {:.4f}, FPS: {:.3f}'.format(epoch,count, loss.data.item(),step_time,FPS))
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  ANN_Accuracy: {} %'.format(count, loss.data.item(), accuracy))
                


# In[7]:


# visualization loss 

plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()


# ## CNN (Convolution Neural Network)

# In[8]:


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


# In[9]:


# Create CNN
model = CNNModel().npu()

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = apex.optimizers.NpuFusedSGD(model.parameters(), lr=learning_rate)
model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O2',
                                      loss_scale='128.0',
                                      combine_grad=True)

# In[10]:


# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 1, 28, 28)).npu()
        labels = Variable(labels).npu()
        
        
        optimizer.zero_grad() # Clear gradients
        outputs = model(train) # Forward propagation
        loss = error(outputs, labels) # Calculate softmax and cross entropy loss
        # loss.backward() # Calculating gradients
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step() # Update parameters
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            
            # Predict test dataset
            for images, labels in test_loader:
                test = Variable(images.view(-1, 1, 28, 28)).npu()
                outputs = model(test) # Forward propagation
                predicted = torch.max(outputs.data, 1)[1].npu() # Get predictions from the maximum value
                total += len(labels) # Total number of labels
                correct += (predicted == labels.npu()).sum() # Total correct predictions
            
            accuracy = 100.0 * correct.item() / total
            
            # store loss and iteration
            loss_list.append(loss.data.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  CNN_Accuracy: {} %'.format(count, loss.data.item(), accuracy))
                


# In[11]:


# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()


# ## Submit

# In[12]:


test = pd.read_csv("../datasets/Kannada-MNIST/test.csv",dtype = np.float32)

index = test['id']
test.drop('id', axis=1, inplace=True)
test = test.values/255 # normalization

test = torch.from_numpy(test)
test = Variable(test.view(-1, 1, 28, 28)).npu()


# In[13]:


res = model(test)
sub = pd.read_csv("../datasets/Kannada-MNIST/sample_submission.csv")
sub['label'] = torch.max(res.data.cpu(), 1)[1]
sub.to_csv('submission.csv', index=False)


# In[ ]:




