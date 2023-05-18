# Copyright 2022 Huawei Technologies Co., Ltd
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

import torch

class PermuteImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):
        ctx.save_for_backward(input_data)
        conv = input_data.squeeze(2)
        result = conv.transpose(1, 2).contiguous().transpose(0, 1)
        return result.detach()

    @staticmethod
    def backward(grad_output):
        out = grad_output.transpose(0, 1).contiguous().transpose(1, 2)
        out = out.unsqueeze(2)
        return out


class BidirectionalLSTM(torch.nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.fw_rnn = torch.nn.LSTM(nIn, nHidden, bidirectional=False)
        self.bw_rnn = torch.nn.LSTM(nIn, nHidden, bidirectional=False)
        self.embedding = torch.nn.Linear(nHidden * 2, nOut)

    def forward(self, input_data):
        input_fw = input_data
        recurrent_fw, _ = self.fw_rnn(input_fw)
        input_bw = torch.flip(input_data, [0])
        recurrent_bw, _ = self.bw_rnn(input_bw)
        recurrent_bw = torch.flip(recurrent_bw, [0])
        recurrent = torch.cat((recurrent_fw, recurrent_bw), 2)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(torch.nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = torch.nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           torch.nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), torch.nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), torch.nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), torch.nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), torch.nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       torch.nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = torch.nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input_data):

        # conv features
        conv = self.cnn(input_data)
        _, _, h, _ = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = torch.nn.functional.log_softmax(self.rnn(conv), dim=2)

        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def squeeze_permute(x):
    return PermuteImplementation.apply(x)


def get_crnn(config):
    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    model.apply(weights_init)

    return model
