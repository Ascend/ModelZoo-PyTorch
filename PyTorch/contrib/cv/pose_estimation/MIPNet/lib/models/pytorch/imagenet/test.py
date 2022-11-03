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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# encoding: utf-8
import torch
 
pthfile = r'./lib/models/pytorch/imagenet/hrnet_w48-8ef0771d.pth'            #.pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))    #设置在cpu环境下查询
print('type:')
print(type(model))  #查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  #查看模型字典里面的key
    print(k)
print('value:')
for k in model:         #查看模型字典里面的value
    print(k,model[k])