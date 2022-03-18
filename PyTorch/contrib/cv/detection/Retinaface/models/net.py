#Copyright 2021 Huawei Technologies Co., Ltd
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
import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

class Conv_Bn(nn.Module):
    def __init__(self, inp, oup, stride=1, leaky=0):
        super(Conv_Bn, self).__init__()
        self.tunnel = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=False)
    )

    def forward(self, x):
        return self.tunnel(x)


class Conv_Bn_No_Relu(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Conv_Bn_No_Relu, self).__init__()
        self.tunnel = nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

    def forward(self,x):
        return self.tunnel(x)


class Conv_Bn1X1(nn.Module):
    def __init__(self, inp, oup, stride, leaky=0):
        super(Conv_Bn1X1, self).__init__()
        self.tunnel = nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=False)
    )

    def forward(self,x):
        return self.tunnel(x)

class Conv_Dw(nn.Module):
    def __init__(self, inp, oup, stride, leaky=0.1):
        super(Conv_Dw, self).__init__()
        self.tunnel = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.LeakyReLU(negative_slope= leaky,inplace=False),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(negative_slope=leaky, inplace=False),
        )

    def forward(self, x):
        return self.tunnel(x)


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = Conv_Bn_No_Relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = Conv_Bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = Conv_Bn_No_Relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = Conv_Bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = Conv_Bn_No_Relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = Conv_Bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = Conv_Bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = Conv_Bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = Conv_Bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = Conv_Bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            Conv_Bn(3, 8, 2, leaky = 0.1),    # 3
            Conv_Dw(8, 16, 1),   # 7
            Conv_Dw(16, 32, 2),  # 11
            Conv_Dw(32, 32, 1),  # 19
            Conv_Dw(32, 64, 2),  # 27
            Conv_Dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            Conv_Dw(64, 128, 2),  # 43 + 16 = 59
            Conv_Dw(128, 128, 1), # 59 + 32 = 91
            Conv_Dw(128, 128, 1), # 91 + 32 = 123
            Conv_Dw(128, 128, 1), # 123 + 32 = 155
            Conv_Dw(128, 128, 1), # 155 + 32 = 187
            Conv_Dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            Conv_Dw(128, 256, 2), # 219 +3 2 = 241
            Conv_Dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

