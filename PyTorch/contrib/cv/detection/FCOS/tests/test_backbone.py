#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import torch
from apex import amp
from mmdet.models.backbones.resnet import ResNet

AMP_MODE = True

model = ResNet(depth=50,
           num_stages=4,
           out_indices=(0, 1, 2, 3),
           frozen_stages=1,
           #norm_cfg=dict(type='BN', requires_grad=False),
           norm_eval=True,
           style='caffe')
x = torch.randn(2,3,800,800)
optimizer = torch.optim.SGD(model.parameters(),0.1)


# 设置hook func
def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        # 请依据使用场景自定义函数
        print(name+' inputs')
        print(name+' outputs')
    return hook_function

# 注册正反向hook
for name, module in model.named_modules():
    module.register_forward_hook(hook_func('[forward]: '+name, module))
    module.register_backward_hook(hook_func('[backward]: '+name, module))

torch.npu.set_device("npu:0")
model = model.npu()
x = x.npu()

if AMP_MODE:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=1.0)

o = model(x)
l1,l2,l3,l4 = [i.mean() for i in o]
l = l1+l2+l3+l4
print(l)
l.backward()
