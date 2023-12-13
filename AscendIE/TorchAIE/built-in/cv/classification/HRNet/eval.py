# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import time
import os
import argparse

import torch
import numpy as np
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from PIL import Image

import torch_aie


def parse_args():
    parser = argparse.ArgumentParser(description="InceptionV3 Sample")
    parser.add_argument('--device_id', type=int, default=0, help="please set device id")
    parser.add_argument('--dataset', type=str, required=True, help="JEPG_dir, /path/to/imagenet/val")
    parser.add_argument('--batch_size', type=int, default=1, help="please set batch_size")
    parser.add_argument('--label_file', type=str, help="imagenet label txt", required=True)
    parser.add_argument('--traced_model', type=str, help="traced_model file", required=True)
    args = parser.parse_args()
    return args


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def get_input_data(jpeg_file, imagenet_dir):
    image = Image.open(os.path.join(imagenet_dir, jpeg_file))
    image = image.convert('RGB')
    image = resize(image, 256)  # Resize
    image = center_crop(image, 224)  # CenterCrop
    img = np.array(image, dtype=np.float32)
    img = img.transpose(2, 0, 1)  # ToTensor: HWC -> CHW
    img = img / 255.  # ToTensor: div 255
    img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]  # Normalize: mean
    img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]  # Normalize: std
    img = torch.from_numpy(img).contiguous()
    return img


class CustomDataset(data.Dataset):
    def __init__(self, args):
        self.prep_data_list = os.listdir(args.dataset)
        self.args = args

    def __getitem__(self, index):
        jpeg_file = self.prep_data_list[index - 1]
        input_tensor = get_input_data(jpeg_file, self.args.dataset)
        return input_tensor, jpeg_file

    def __len__(self):
        return len(self.prep_data_list)


class HRNet:

    def __init__(self, args):
        self.args = args
        self.topn = 5
        self.count_hit = {"top1": 0, "top5": 0}
        self.count = 0
        self.img_gt_dict = {}
        self.device_name = "npu:" + str(self.args.device_id)
        self.stream = torch_aie.npu.Stream(self.device_name)
        self.traced_model = None
        self.compiled_model_bs1 = None

    def load_trace_model(self):
        self.traced_model = torch.jit.load(self.args.traced_model)
        print("load trace model success.")

    def create_label(self):
        img_gt_dict = {}
        with open(self.args.label_file, 'r') as f:
            for line in f.readlines():
                temp = line.strip().split(" ")
                imgName = temp[0].split(".")[0]
                imgLab = temp[1]
                img_gt_dict[imgName] = imgLab
        self.img_gt_dict = img_gt_dict
        print("create labels success.")

    def compile_model(self, batch_size=1):
        if self.traced_model is None:
            self.load_trace_model()
        input_info = [torch_aie.Input((args.batch_size, 3, 224, 224))]
        print('Start compiling model.')
        compiled_model = torch_aie.compile(
            self.traced_model,
            inputs=input_info,
            precision_policy=torch_aie.PrecisionPolicy.FP16,
            allow_tensor_replace_int=True,
            soc_version="Ascend310P3")
        print('Model compiled successfully.')
        if batch_size == 1:
            self.compiled_model_bs1 = compiled_model
        return compiled_model

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

    def cal_accuracy(self):
        print("start compute accuracy ... ")
        self.create_label()
        if self.compiled_model_bs1 is None:
            self.compiled_model_bs1 = self.compile_model()
        dataset = CustomDataset(self.args)
        dataloader = torch.utils.data.DataLoader(dataset, 1, drop_last=True, shuffle=False)
        for index, (data, jpeg_file) in tqdm(enumerate(dataloader)):
            data = data.to(self.device_name)
            ret = self.compiled_model_bs1(data).to("cpu")
            self.update_hit_cnt(ret, list(jpeg_file)[0])
            self.count += 1
        print("top1", self.count_hit["top1"] / self.count)
        print("top5", self.count_hit["top5"] / self.count)

    def cal_qps(self, model, batch_size=1, warm_up=5):
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
    net = HRNet(args)
    model = net.compile_model(args.batch_size)
    net.cal_accuracy()
    net.cal_qps(model, args.batch_size)
