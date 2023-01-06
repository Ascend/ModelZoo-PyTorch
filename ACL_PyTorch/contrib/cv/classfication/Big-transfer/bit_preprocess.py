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
from tqdm import tqdm
import torchvision as tv
import torch.utils.data
import torch.nn.functional as F

def preprocess(dataset_path, data_bin_path, label_path, batch_size):

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((128, 128)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    valid_set = tv.datasets.CIFAR10(dataset_path, transform=val_tx, train=False, download=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False)

    if not os.path.isdir(data_bin_path):
        os.mkdir(data_bin_path)

    with open(label_path, 'x') as f:
        for i, (images, target) in tqdm(enumerate(valid_loader)):
            label = ' '.join((str(i) for i in target.tolist()))
            f.write(label+'\n')
            save_file_name = "{}.bin".format(i)
            save_path = os.path.join(data_bin_path, save_file_name)
            if images.shape[0] != batch_size:
                images = F.pad(input=images, pad=(0, 0, 0, 0, 0, 0, 0, batch_size-images.shape[0]), mode='constant', value=0)
            images.numpy().tofile(save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='bit_preprocess')
    parser.add_argument('--dataset_path', type=str, help='dataset path', required=True)
    parser.add_argument('--save_path', type=str, help='bin file save path', required=True)
    parser.add_argument('--label_path', type=str, help='path to save label', required=True)
    parser.add_argument('--batch_size', type=int, default=1, help='om batch size')
    args = parser.parse_args()

    preprocess(args.dataset_path, args.save_path, args.label_path, args.batch_size)