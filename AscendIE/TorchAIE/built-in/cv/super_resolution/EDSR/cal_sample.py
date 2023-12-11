# Copyright 2023 Huawei Technologies Co., Ltd
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

import sys

sys.path.append(r'./EDSR-PyTorch/src')
import model
import utility

import torch
from torch.utils import data
import numpy as np
import os
import argparse
import json
import math
import imageio
from tqdm import tqdm
import time
import torch_aie


def get_args():
    parser = argparse.ArgumentParser(description='EDSR Sample script')
    parser.add_argument('--HR', type=str, help='high target image path')
    parser.add_argument('--prep_data', type=str, help='prep_data bin path')
    parser.add_argument('--pth', default='./edsr_baseline_x2-1bc95232.pt', type=str, help='/path/to/pth')
    parser.add_argument("--pad_info", default="pad_info.json", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--save", default="results", type=str)
    args = parser.parse_args()
    return args


class CustomDataset(data.Dataset):
    def __init__(self, args):
        self.prep_data_list = os.listdir(args.prep_data)
        self.args = args

    def __getitem__(self, index):
        bin_file = self.prep_data_list[index - 1]
        input_tensor = torch.from_numpy(np.fromfile(os.path.join(self.args.prep_data, bin_file), dtype=np.float32)) \
            .reshape(3, 1020, 1020)
        return input_tensor, input_tensor

    def __len__(self):
        return len(self.prep_data_list)


class Margs:
    def __init__(self):
        self.scale = [2]
        self.pre_train = args.pth
        self.test_only = True
        self.cpu = True
        self.load = ''
        self.save = ''
        self.reset = ''
        self.data_test = ''
        self.model = 'EDSR'
        self.self_ensemble = False
        self.chop = False
        self.precision = 'single'
        self.n_GPUs = 1
        self.save_models = False
        self.n_resblocks = 16
        self.n_feats = 64
        self.rgb_range = 255
        self.n_colors = 3
        self.res_scale = 1
        self.resume = 0


class TestEDSR:
    def __init__(self, args):
        self.args = args
        self.device_name = "npu:" + str(args.device_id)
        self.stream = torch_aie.npu.Stream(self.device_name)
        self.scale = 2
        self.size = 1020
        self.src_model = None
        self.compiled_torch_aie_bs1 = None

    def cal_qps(self, model, batch_size, warm_up=5):
        cost_steps = []
        dataset = CustomDataset(args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last = True, shuffle=False)
        for index, (data, _) in enumerate(dataloader):
            print("data.shape", data.shape)
            data = data.to(self.device_name)
            with torch_aie.npu.stream(self.stream):
                ts = time.time()
                pre_hr = model(data)
                self.stream.synchronize()
                te = time.time()
                cost_steps.append(te - ts)
        print("QPS is {}, bs is {}".format(
            batch_size * (len(cost_steps) - warm_up) / sum(cost_steps[warm_up:-1]), batch_size))

    def get_pad_info(self):
        with open(self.args.pad_info) as f:
            pad_info = json.load(f)
        return pad_info

    def compile(self, batch_size=1):
        if batch_size == 1 and self.compiled_torch_aie_bs1 is not None:
            print("model has been compiled succss, batch size is ", batch_size)
            return self.compiled_torch_aie_bs1
        checkpoint = utility.checkpoint(Margs())
        src_model = model.Model(Margs(), checkpoint)
        src_model.load_state_dict(torch.load(
            self.args.pth, map_location=torch.device('cpu')), strict=False)
        src_model.eval()
        dummy_input = torch.randn(1, 3, 1020, 1020)
        traced_model = torch.jit.trace(src_model, dummy_input)
        torch_aie_model = torch_aie.compile(traced_model, inputs=[torch_aie.Input([batch_size, 3, 1020, 1020])],
                                            precision_policy=torch_aie.PrecisionPolicy.FP16)
        if batch_size == 1:
            self.compiled_torch_aie_bs1 = torch_aie_model
        print("compile model success, batch size is ", batch_size)
        return torch_aie_model

    def cal_accuracy(self):
        psnr_data = []
        for file in tqdm(os.listdir(self.args.prep_data)):
            if file.endswith(".bin"):
                input_tensor = torch.from_numpy(np.fromfile(os.path.join(self.args.prep_data, file), dtype=np.float32)) \
                    .reshape(1, 3, 1020, 1020).to(self.device_name)
                if self.compiled_torch_aie_bs1 is None:
                    self.compile(batch_size=1)
                with torch_aie.npu.stream(self.stream):
                    ts = time.time()
                    pre_hr = self.compiled_torch_aie_bs1(input_tensor)
                    self.stream.synchronize()
                    te = time.time()
                pre_hr = pre_hr.to("cpu")
                pre_hr = pre_hr.reshape(3, self.size * self.scale, self.size * self.scale)
                pre_hr = pre_hr.permute(1, 2, 0)
                pre_hr = self.quantize(pre_hr, 255)
                pre_hr = self.crop(file, pre_hr)
                for img_file in os.listdir(self.args.HR):
                    if img_file[0:4] in file:
                        tgt_hr = imageio.imread(os.path.join(self.args.HR, img_file))
                        tgt_hr = torch.from_numpy(tgt_hr)
                        psnr = self.calc_psnr(pre_hr, tgt_hr, self.scale, 255)
                        psnr_data.append({"file": file, "psnr": psnr})

        psnr_data = self.eval_acc(psnr_data)
        json_data = json.dumps(psnr_data, indent=4, separators=(',', ': '))
        with open("result.json", 'w') as f:
            f.write(json_data)

    def crop(self, file, pre_hr):
        pad_info = self.get_pad_info()
        for pad_meta in pad_info:
            if file[0:4] in pad_meta['name']:
                pad_x = pad_meta['pad_x'] * self.scale
                pad_y = pad_meta['pad_y'] * self.scale

        if pad_x == 0 and pad_y == 0:
            return pre_hr
        elif pad_x == 0:
            pre_hr = pre_hr[0:-pad_y, :, :]
        elif pad_y == 0:
            pre_hr = pre_hr[:, 0:-pad_x, :]
        else:
            pre_hr = pre_hr[0:-pad_y, 0:-pad_x, :]
        return pre_hr

    def eval_acc(self, data):
        acc = 0
        print(len(data))
        for item in data:
            acc += item["psnr"]
        acc /= len(data)
        print("accuracy: ", acc)
        return {
            "accuracy": acc,
            "data": data
        }

    def quantize(self, img, rgb_range):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    def calc_psnr(self, sr, hr, scale, rgb_range):
        sr = sr.permute(2, 0, 1).unsqueeze(0)
        hr = hr.permute(2, 0, 1).unsqueeze(0)
        if hr.nelement() == 1:
            return 0
        diff = (sr - hr) / rgb_range
        shave = scale + 6

        valid = diff[..., shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()

        return -10 * math.log10(mse)


if __name__ == '__main__':

    args = get_args()
    torch_aie.set_device(args.device_id)
    edsr_sample = TestEDSR(args)
    model = edsr_sample.compile(args.batch_size)
    edsr_sample.cal_accuracy()
    edsr_sample.cal_qps(model, args.batch_size)