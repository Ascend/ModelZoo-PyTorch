# -*- coding: utf-8 -*-
"""
Copyright 2021 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.onnx
import torch._utils
from collections import OrderedDict


class BidirectionalLSTM(torch.nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.fw_rnn = torch.nn.LSTM(nIn, nHidden, bidirectional=False)
        self.bw_rnn = torch.nn.LSTM(nIn, nHidden, bidirectional=False)
        self.embedding = torch.nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        input_fw = input
        recurrent_fw, _ = self.fw_rnn(input_fw)
        input_bw = torch.flip(input, [0])
        recurrent_bw, _ = self.bw_rnn(input_bw)
        recurrent_bw = torch.flip(recurrent_bw, [0])
        recurrent = torch.cat((recurrent_fw, recurrent_bw), 2)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        # [T * b, nOut]
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output


class CRNN(torch.nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
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
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               torch.nn.LeakyReLU(0.2, inplace=True))
            else:
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

    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)  # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = torch.nn.functional.log_softmax(self.rnn(conv), dim=2)

        return output


def proc_node_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def lstm_op_adapter(state_dict):
    ret = {}
    for key, value in state_dict.items():
        if not (key.startswith('rnn') and ('fw' in key or 'bw' in key)):
            ret[key] = value
            continue
        param = state_dict[key].data.split(256)
        ret[key] = torch.cat((param[0], param[2], param[1], param[3]), 0)
    return ret


def convert(pth_file_path, onnx_path):
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    checkpoint['state_dict'] = proc_node_module(checkpoint, 'state_dict')
    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(lstm_op_adapter(checkpoint['state_dict']))
    model.eval()
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(1, 1, 32, 100)
    dynamic_axes = {"actual_imput_1": {0: "-1"}, "output1": {1: "-1"}}
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11)


if __name__ == "__main__":
    pth_file_path = "checkpoint_6_acc_0.7887.pth"
    onnx_path = "crnn_npu_dy.onnx"
    convert(pth_file_path, onnx_path)
