# -*- coding: UTF-8 -*-

'''
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py
     
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num
    
class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''
    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class H_swish(nn.Module):
    '''
    hard swish
    '''
    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y

class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''
    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, BN_momentum):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'
        
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if exp_size == in_channels_num:
            # Without expansion, the first depthwise convolution is omitted
            self.conv1 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )
        else:
            # With expansion
            self.conv1 = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=exp_size, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )

    def forward(self, x, expand=False):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        if self.use_residual:
            out = out + x
        if expand:
            return out, out1
        else:
            return out

class MobileNetV3(nn.Module):
    '''
    
    '''
    def __init__(self, mode='small', classes_num=1000, input_size=224, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False):
        '''
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(MobileNetV3, self).__init__()

        mode = mode.lower()
        assert mode in ['large', 'small']
        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100 or Tiny-ImageNet
            s = 1

        # setting of the model
        if mode == 'large':
            # Configuration of a MobileNetV3-Large Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', s],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        elif mode == 'small':
            # Configuration of a MobileNetV3-Small Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, 'RE', s],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        first_channels_num = 16

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1280 if mode == 'large' else 1024

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum),
            H_swish()
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL, BN_momentum))
            input_channels_num = output_channels_num
        
        # the last stage
        last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        last_stage_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channels_num, out_channels=last_stage_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_stage_channels_num, momentum=BN_momentum),
                H_swish()
            )
        feature_extraction_layers.append(last_stage_layer1)

        
        self.featureList = nn.ModuleList(feature_extraction_layers)

        # SE Module
        # remove the last SE Module according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # feature_extraction_layers.append(SEModule(last_stage_channels_num) if mode == 'small' else nn.Sequential())

        last_stage = []
        last_stage.append(nn.AdaptiveAvgPool2d(1))
        last_stage.append(nn.Conv2d(in_channels=last_stage_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1, padding=0, bias=False))
        last_stage.append(H_swish())

        self.last_stage_layers = nn.Sequential(*last_stage)
        

        ########################################################################################################################
        # Classification part
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels_num, classes_num)
        )
        

        '''
        self.extras = nn.ModuleList([
            InvertedResidual(576, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])
        '''

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        for i in range(9):
            x = self.featureList[i](x)
        x = self.featureList[9](x)
        for i in range(10, len(self.featureList)):
            x = self.featureList[i](x)
        x = self.last_stage_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
	            if hasattr(m, 'lastBN'):
	                nn.init.constant_(m.lastBN.weight, 0.0)

if __name__ == "__main__":
    width_multiplier = 1
    from torchsummaryX import summary
    
    # cifar10
    #model_large = MobileNetV3(mode='large', classes_num=10, input_size=32, width_multiplier=width_multiplier)
    #model_large.eval()
    '''
    model_small = MobileNetV3(mode='small', classes_num=10, input_size=32, width_multiplier=width_multiplier)
    model_small.eval()
    summary(model_small, torch.zeros((1, 3, 32, 32)))
    print('MobileNetV3-Small-%.2f cifar10-summaryX\n' % width_multiplier)
    '''
    '''
    # cifar100
    model_large = MobileNetV3(mode='large', classes_num=100, input_size=32, width_multiplier=width_multiplier)
    model_large.eval()
    model_small = MobileNetV3(mode='small', classes_num=100, input_size=32, width_multiplier=width_multiplier)
    model_small.eval()
    
    input = torch.randn(1, 3, 32, 32)
    from thop import profile
    FLOPs_large, params_large = profile(model_large, inputs=(input,))
    FLOPs_small, params_small = profile(model_small, inputs=(input,))

    print('\nOn cifar100 using thop')
    print('MobileNetV3-Large-%.2f:' % width_multiplier)
    print('Total flops: %.4fM' % (FLOPs_large/1000000.0))
    print('Total params: %.4fM' % (params_large/1000000.0))
    print()
    print('MobileNetV3-Small-%.2f:' % width_multiplier)
    print('Total flops: %.4fM' % (FLOPs_small/1000000.0))
    print('Total params: %.4fM' % (params_small/1000000.0))
    
    summary(model_large, torch.zeros((1, 3, 32, 32)))
    print('MobileNetV3-Large-%.2f cifar100-summaryX\n' % width_multiplier)
    summary(model_small, torch.zeros((1, 3, 32, 32)))
    print('MobileNetV3-Small-%.2f cifar100-summaryX\n' % width_multiplier)
    
    # ImageNet
    model_large = MobileNetV3(mode='large', classes_num=1000, input_size=224, width_multiplier=width_multiplier)
    model_large.eval()
    model_small = MobileNetV3(mode='small', classes_num=1000, input_size=224, width_multiplier=width_multiplier)
    model_small.eval()
    
    
    input = torch.randn(1, 3, 224, 224)
    from thop import profile
    FLOPs_large, params_large = profile(model_large, inputs=(input,))
    FLOPs_small, params_small = profile(model_small, inputs=(input,))

    print('\nOn ImageNet using thop')
    print('MobileNetV3-Large-%.2f:' % width_multiplier)
    print('Total flops: %.4fM' % (FLOPs_large/1000000.0))
    print('Total params: %.4fM' % (params_large/1000000.0))
    print()
    print('MobileNetV3-Small-%.2f:' % width_multiplier)
    print('Total flops: %.4fM' % (FLOPs_small/1000000.0))
    print('Total params: %.4fM' % (params_small/1000000.0))
    '''
    '''
    summary(model_large, torch.zeros((1, 3, 224, 224)))
    print('MobileNetV3-Large-%.2f ImageNet-summaryX\n' % width_multiplier)
    summary(model_small, torch.zeros((1, 3, 224, 224)))
    print('MobileNetV3-Small-%.2f ImageNet-summaryX\n' % width_multiplier)
    '''