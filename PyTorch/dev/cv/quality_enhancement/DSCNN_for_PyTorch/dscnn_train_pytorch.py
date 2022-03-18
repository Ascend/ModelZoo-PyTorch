# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the MIT License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torch.npu


class DataLoad():
    def __init__(self, src_data_root, label_path):
        self.images_list = []
        self.label_list = []
        self.num = 0

        for img_name in os.listdir(src_data_root):
            if img_name.endswith(".png"):
                src_img_path = os.path.join(src_data_root, img_name)
                label_img_path = os.path.join(label_path, img_name)
                assert os.path.exists(label_img_path)
                self.images_list.append([src_img_path])
                self.label_list.append([label_img_path])
                self.num += 1
        print('train image num: ', self.num)

    def __getitem__(self, index):

        src_image = Image.open(self.images_list[index][0])
        src_image = np.asarray(src_image).astype(np.float32) / 255.
        label_image = Image.open(self.label_list[index][0])
        label_image = np.asarray(label_image).astype(np.float32) / 255.


        src_image = torch.from_numpy(src_image.transpose(2, 0, 1))
        label_image = torch.from_numpy(label_image.transpose(2, 0, 1))

        return src_image, label_image

    def __len__(self):
        return self.num


class downsample_net(nn.Module):
    def __init__(self):
        super(downsample_net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 3, 3, 1, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x)



def train(src_data_root, label_path, batch_size, model_dir, epoch, lr, device, model_path=None):
    dn = downsample_net()
    dn = dn.to(device)

    if model_path is not None:
        dn.load_state_dict(torch.load(model_path))

    l1loss = torch.nn.L1Loss().to(device)

    opt = torch.optim.Adam(dn.parameters(), lr=lr)

    dataset = DataLoad(src_data_root, label_path)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16)

    all_loss = []
    for ep in range(epoch):
        for step, (sample, label) in enumerate(train_loader):
            src_image = Variable(sample).to(device)
            label_image = Variable(label).to(device)
            out = dn(src_image)

            out = nn.functional.interpolate(out, size=[1080, 1920], mode="bilinear")
            loss = l1loss(out, label_image)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 50 == 0:
                print("epoch {}  step {}  loss {}".format(ep, step, loss.detach()))


        
        model_path = os.path.join(model_dir, "DSCNN_pytorch_l1_" + str(ep) + ".pkl")
        torch.save(dn.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--label_data_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--model_save_dir', type=str, default='')
    parser.add_argument('--pre_trained_model_path', type=str, default=None)
    args = parser.parse_args()

    device_str = 'npu:' + str(args.device_id)
    torch.npu.set_device(device_str)

    train(device=device_str,
        src_data_root=args.train_data_path,
        batch_size=args.batch_size,
        model_dir=args.model_save_dir,
        label_path=args.label_data_path,
        model_path=args.pre_trained_model_path,
        epoch=args.epoch,
        lr=args.lr)
