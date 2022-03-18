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

import torch.nn.functional as F
import math

# from tensorboardX import SummaryWriter


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
	def __init__(self):
		super(MobileNetV3_Large, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.hs1 = hswish()

		self.bneck1 = nn.Sequential(
			Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
			Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
			Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
			nn.Conv2d(24, 72, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(72),
			nn.ReLU(inplace=True),
		)  # 1 72 128 128

		self.bneck2 = nn.Sequential(
			nn.Conv2d(72, 72, kernel_size=5, stride=2, padding=2, groups=72, bias=False),
			nn.BatchNorm2d(72),
			nn.ReLU(inplace=True),
			nn.Conv2d(72, 40, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(40),
			SeModule(40),

			Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
			Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
			nn.Conv2d(40, 240, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(240),
			hswish(),
		)  # 1 240 64 64

		self.bneck3 = nn.Sequential(
			nn.Conv2d(240, 240, kernel_size=3, stride=2, padding=1, groups=240, bias=False),
			nn.BatchNorm2d(240),
			hswish(),
			nn.Conv2d(240, 80, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(80),
			Block(3, 80, 200, 80, hswish(), None, 1),
			Block(3, 80, 184, 80, hswish(), None, 1),
			Block(3, 80, 184, 80, hswish(), None, 1),
			Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
			Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
			Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
			nn.Conv2d(160, 672, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(672),
			hswish(),
		)  # 1 672 32 32

		self.bneck4 = nn.Sequential(
			nn.Conv2d(672, 672, kernel_size=5, stride=2, padding=2, groups=672, bias=False),
			nn.BatchNorm2d(672),
			hswish(),
			nn.Conv2d(672, 160, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(160),
			SeModule(160),
			Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
		)  # 1 160 16 16

		self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(960)
		self.hs2 = hswish()

		self.conv3 = nn.Conv2d(960, 640, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(640)
		self.linear = nn.ReLU(inplace=True)

		self.init_params()

	def init_params(self):
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

	def forward(self, x):
		out = self.hs1(self.bn1(self.conv1(x)))
		# print(out.shape) torch.Size([2, 16, 256, 256])
		out1 = self.bneck1(out)
		# print(out1.shape)  torch.Size([2, 72, 128, 128]) up
		out2 = self.bneck2(out1)
		# print(out2.shape)  torch.Size([2, 240, 64, 64]) up
		out3 = self.bneck3(out2)
		# print(out3.shape)  torch.Size([2, 672, 32, 32]) up
		out = self.bneck4(out3)
		# print(out4.shape) torch.Size([2, 160, 16, 16])
		out = self.hs2(self.bn2(self.conv2(out)))
		# print(out.shape)  torch.Size([2, 960, 16, 16]) up
		out = self.linear(self.bn3(self.conv3(out)))
		# print(out.shape) torch.Size([2, 640, 16, 16])
		return out ,out1,out2,out3,

class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		self.conv1 = nn.Conv2d(1312, 320, 1)
		self.bn1 = nn.BatchNorm2d(320)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(320, 320, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(320)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(560, 160, 1)
		self.bn3 = nn.BatchNorm2d(160)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(160, 160, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(160)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(232, 128, 1)
		self.bn5 = nn.BatchNorm2d(128)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(128)
		self.relu6 = nn.ReLU()

		# self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		# self.bn7 = nn.BatchNorm2d(32)
		# self.relu7 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x,x1,x2,x3):
		# print(x.shape) 1 640 16 16
		y = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
		# print(y.shape) 1 640 32 32
		y = torch.cat((y, x3), 1) # 1 1312 32 32
		y = self.relu1(self.bn1(self.conv1(y)))
		y = self.relu2(self.bn2(self.conv2(y)))  # 1 320 32 32

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) #1 320 64 64
		y = torch.cat((y, x2), 1)   #1 560 64 64
		y = self.relu3(self.bn3(self.conv3(y)))
		y = self.relu4(self.bn4(self.conv4(y)))  # 1 160 64 64

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)  #1 160 128 128
		y = torch.cat((y, x1), 1)  # 1 232 128 128
		y = self.relu5(self.bn5(self.conv5(y)))
		y = self.relu6(self.bn6(self.conv6(y))) # 1 128 128 128

		# y = self.relu7(self.bn7(self.conv7(y)))
		return y

class output(nn.Module):
	def __init__(self):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(128, 1, 1)

		self.conv2 = nn.Conv2d(128, 2, 1)

		self.conv3 = nn.Conv2d(128, 4, 1)


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		inside_score = self.conv1(x)
		side_v_code  = self.conv2(x)
		side_v_coord = self.conv3(x)
		east_detect   = torch.cat((inside_score, side_v_code,side_v_coord), 1)
		return east_detect
		
	
class EAST(nn.Module):
	def __init__(self):
		super(EAST, self).__init__()
		self.extractor = MobileNetV3_Large()
		self.merge     = merge()
		self.output    = output()
	
	def forward(self, x):

		x,x1,x2,x3=self.extractor(x)

		return self.output(self.merge(x,x1,x2,x3))
		

if __name__ == '__main__':

	m = EAST()
	# x = torch.randn(1, 3, 512, 512)
	# with SummaryWriter(comment='mobilenetv3') as w:
	# 	w.add_graph(m, (x,))
	# east_detect= m(x)
	# print(east_detect.shape)
	print(m)

