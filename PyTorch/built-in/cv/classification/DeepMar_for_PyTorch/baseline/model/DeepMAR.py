# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50


class DeepMAR_ResNet50(nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super(DeepMAR_ResNet50, self).__init__()
        # init the necessary parameter for netwokr structure
        if 'num_att' in kwargs:
            self.num_att = kwargs['num_att'] 
        else:
            self.num_att = 35
        if 'last_conv_stride' in kwargs:
            self.last_conv_stride = kwargs['last_conv_stride']
        else:
            self.last_conv_stride = 2
        if 'drop_pool5' in kwargs:
            self.drop_pool5 = kwargs['drop_pool5']
        else:
            self.drop_pool5 = True 
        if 'drop_pool5_rate' in kwargs:
            self.drop_pool5_rate = kwargs['drop_pool5_rate']
        else:
            self.drop_pool5_rate = 0.5
        if 'pretrained' in kwargs:
            self.pretrained = kwargs['pretrained'] 
        else:
            self.pretrained = True
        
        self.base = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        
        self.classifier = nn.Linear(2048, self.num_att)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        # x = F.avg_pool2d(x, x.shape[2:])
        x = F.avg_pool2d(x, (7, 7))
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        return x

class DeepMAR_ResNet50_ExtractFeature(object):
    """
    A feature extraction function
    """
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_model = self.model.training

        # set the model to be eval
        self.model.eval()

        # imgs should be Variable
        if not isinstance(imgs, Variable):
            print ('imgs should be type: Variable')
            raise ValueError
        # compute output
        score = self.model(imgs)
        score = score.data.cpu().numpy()

        # set the model to be training
        self.model.train(old_train_eval_model)

        return score
