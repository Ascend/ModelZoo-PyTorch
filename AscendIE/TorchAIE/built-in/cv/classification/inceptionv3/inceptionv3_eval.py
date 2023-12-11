# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
from PIL import Image
import numpy as np
import argparse
import multiprocessing
import torch
import torch_aie
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="InceptionV3 Sample")
    parser.add_argument('--device_id', type=int, default = 0, help="please set device id")
    parser.add_argument('--prep_dataset', type=str, required=True, help="please set prepocess data dir")
    parser.add_argument('--batch_size', type=int, default = 1, help="please set batch_size")
    parser.add_argument('--label_file', type=str, help="imagenet label txt", required=True)
    parser.add_argument('--checkpoint', type=str, help="checkpoint path", required=True)
    args = parser.parse_args()
    return args

class CustomDataset(data.Dataset):
    def __init__(self, args):
        self.prep_data_list = os.listdir(args.prep_dataset)
        self.args = args

    def __getitem__(self, index):
        bin_file = self.prep_data_list[index - 1]
        input_tensor = torch.from_numpy(np.fromfile(os.path.join(self.args.prep_dataset, bin_file), dtype=np.float32)) \
            .reshape([3, 299, 299])
        return input_tensor, bin_file

    def __len__(self):
        return len(self.prep_data_list)

class Inceptionv3:

    def __init__(self, args):
        self.args = args
        self.img_gt_dict = {}
        self.device_name = "npu:" + str(args.device_id)
        self.stream = torch_aie.npu.Stream(self.device_name)
        self.count_hit = {"top1": 0, "top5": 0}
        self.count = 0
        self.compiled_model_bs1 = None
        self.traced_model = None
        self.topn = 5

    def create_label(self):
        img_gt_dict = {}
        with open(self.args.label_file, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split(" ")
                imgName = temp[0].split(".")[0]
                imgLab = temp[1]
                img_gt_dict[imgName] = imgLab
        self.img_gt_dict = img_gt_dict

    def read_bin(self, bin_file):
        data = np.fromfile(os.path.join(self.args.prep_dataset, bin_file),
                            dtype=np.float32).reshape(3, 299, 299)
        data = torch.from_numpy(data).unsqueeze(0).to(self.device_name)  # [1, 3, 299, 299]
        return data

    def trace_model(self):
        print("start trace model ...")
        model = models.inception_v3(pretrained=False, transform_input=True, init_weights=False)
        checkpoint = torch.load(self.args.checkpoint, map_location=None)
        model.load_state_dict(checkpoint)
        model.eval()
        dummy_input = torch.randn(1, 3, 299, 299)
        ts_model = torch.jit.trace(model, dummy_input)
        self.traced_model = ts_model
        print("trace model success")

    def compile_model(self, batch_size = 1):
        if self.traced_model is None:
            self.trace_model()
        input_info = [torch_aie.Input((batch_size, 3, 299, 299))]
        compiled_model = torch_aie.compile(self.traced_model, inputs = input_info,
                                           precision_policy = torch_aie.PrecisionPolicy.FP16,
                                           soc_version="Ascend310P3")
        print("compiled model success, bs is ", batch_size)
        return compiled_model

    def cal_accuracy(self):
        print("strat calculate accuracy...")
        if len(self.img_gt_dict) == 0:
            self.create_label()
        if self.compiled_model_bs1 is None:
            self.compiled_model_bs1 = self.compile_model()
        for bin_file in tqdm(os.listdir(self.args.prep_dataset)):
            data = self.read_bin(bin_file)
            ret = self.compiled_model_bs1(data).to("cpu")
            self.update_hit_cnt(ret, bin_file)
            self.count += 1
        print(self.count)
        print(self.count_hit)
        print("top1:", self.count_hit["top1"] / self.count)
        print("top5:", self.count_hit["top5"] / self.count)

    def update_hit_cnt(self, infer_ret, jpeg_file):
        prediction = np.array(infer_ret.numpy()).reshape(1000)
        n_labels = infer_ret.shape[1]
        sort_index = prediction.argsort()
        realLabel = int(self.img_gt_dict[jpeg_file.split(".")[0]])
        resCnt = []
        for i in range(1, self.topn + 1):
            resCnt.append(sort_index[-i])
        if realLabel == resCnt[0]:
            self.count_hit["top1"] += 1
            self.count_hit["top5"] += 1
        elif realLabel in resCnt:
            self.count_hit["top5"] += 1

    def cal_qps(self, model, batch_size = 1, warm_up=5):
        print("start calculate qps...")
        cost_steps = []
        dataset = CustomDataset(self.args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=False)
        for index, (data, _) in tqdm(enumerate(dataloader)):
            data = data.to(self.device_name)
            with torch_aie.npu.stream(self.stream):
                ts = time.time()
                _ = model(data)
                self.stream.synchronize()
                te = time.time()
                cost_steps.append(te - ts)
        print("QPS is {}, bs is {}".format(
            batch_size * (len(cost_steps) - warm_up) / sum(cost_steps[warm_up:-1]), batch_size))

if __name__ == '__main__':
    args = parse_args()
    torch_aie.set_device(args.device_id)
    inceptionv3 = Inceptionv3(args)
    model = inceptionv3.compile_model(args.batch_size)
    inceptionv3.cal_accuracy()
    inceptionv3.cal_qps(model, args.batch_size)