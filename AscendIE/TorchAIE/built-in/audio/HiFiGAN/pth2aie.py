# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import argparse
import json
import torch
import torch_aie
from env import AttrDict
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ConvTranspose2d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=dilation[0],
                               padding=(0, get_padding(kernel_size, dilation[0])))),
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=dilation[1],
                               padding=(0, get_padding(kernel_size, dilation[1])))),
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=dilation[2],
                               padding=(0, get_padding(kernel_size, dilation[2]))))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=1,
                               padding=(0, get_padding(kernel_size, 1)))),
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=1,
                               padding=(0, get_padding(kernel_size, 1)))),
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=1,
                               padding=(0, get_padding(kernel_size, 1))))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=dilation[0],
                               padding=(0, get_padding(kernel_size, dilation[0])))),
            weight_norm(Conv2d(channels, channels, (1, kernel_size), 1, dilation=dilation[1],
                               padding=(0, get_padding(kernel_size, dilation[1]))))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv2d(80, h.upsample_initial_channel, (1, 7), 1, padding=(0, 3)))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose2d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                (1, k), (1, u), padding=(0, (k - u) // 2))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv2d(ch, 1, (1, 7), 1, padding=(0, 3)))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def adjust_checkpoint(checkpoint):
    for weight in ['weight_g', 'weight_v']:
        for sub_layer in ['conv_pre', 'ups.0', 'ups.1', 'ups.2', 'ups.3', 'conv_post']:
            checkpoint[f'{sub_layer}.{weight}'] = checkpoint[f'{sub_layer}.{weight}'].unsqueeze(2)

        for i in range(12):
            for j in range(3):
                checkpoint[f'resblocks.{i}.convs1.{j}.{weight}'] = checkpoint[
                    f'resblocks.{i}.convs1.{j}.{weight}'].unsqueeze(2)
                checkpoint[f'resblocks.{i}.convs2.{j}.{weight}'] = checkpoint[
                    f'resblocks.{i}.convs2.{j}.{weight}'].unsqueeze(2)

    return checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mel_len', type=int, default=250)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--checkpoint_file', type=str, default='generator_v1')
    parser.add_argument('--config_file', type=str, default='config_v1.json')
    parser.add_argument('--aie_dir', type=str, default='./aie_model.ts')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--dynamic_dim', type=bool, default=False)
    args = parser.parse_args()

    # load config
    with open(args.config_file) as f:
        data = f.read()
    json_config = json.loads(data)

    # load model
    generator = Generator(AttrDict(json_config))
    state_dict_g = torch.load(args.checkpoint_file, map_location='cpu')
    state_dict_g['generator'] = adjust_checkpoint(state_dict_g['generator'])
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # export ts
    print("Starting trace hifigan……")
    dummy_input = torch.randn(args.batch_size, 80, 1, args.mel_len)
    traced_model = torch.jit.trace(generator, (dummy_input), strict=False)
    print("Trace hifigan success.")
    input_info = [torch_aie.Input((args.batch_size, 80, 1, args.mel_len))]
    print("Starting export hifigan to aie……")
    torch_aie.set_device(args.device_id)
    pt_model = torch_aie.compile(
        traced_model,
        inputs=input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        optimization_level=0,
        soc_version="Ascend310P3")

    print("export hifi-gan to aie success")
    pt_model.save(args.aie_dir)
    print("save aie model success")

    if args.dynamic_dim:
        input_info_dim1 = [torch_aie.Input((args.batch_size, 80, 1, 250))]
        input_info_dim2 = [torch_aie.Input((args.batch_size, 80, 1, 500))]
        input_info_dim3 = [torch_aie.Input((args.batch_size, 80, 1, 750))]
        input_info_dim4 = [torch_aie.Input((args.batch_size, 80, 1, 1000))]
        input_info_dim5 = [torch_aie.Input((args.batch_size, 80, 1, 1250))]
        input_info_dim6 = [torch_aie.Input((args.batch_size, 80, 1, 1500))]
        input_info_dim7 = [torch_aie.Input((args.batch_size, 80, 1, 1750))]
        input_info_dim8 = [torch_aie.Input((args.batch_size, 80, 1, 2000))]
        input_info_all_dim = [input_info_dim1, input_info_dim2, input_info_dim3, input_info_dim4,
                              input_info_dim5, input_info_dim6, input_info_dim7, input_info_dim8]
        pt_model_all_dim = torch_aie.compile(
            traced_model,
            inputs=input_info_all_dim,
            precision_policy=torch_aie.PrecisionPolicy.FP16,
            allow_tensor_replace_int=True,
            optimization_level=0,
            soc_version="Ascend310P3")
        pt_model_all_dim.save("bs" + str(args.batch_size) + "_aie_all_dim.pt")
        print("save dynamic dim aie model success")

