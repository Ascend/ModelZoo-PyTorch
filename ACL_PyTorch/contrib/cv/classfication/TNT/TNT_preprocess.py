# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.models import create_model
import tnt
import torch
import argparse

data_config = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (
    0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.9}
mean = torch.tensor([x * 255 for x in data_config['mean']]).view(1, 3, 1, 1)
std = torch.tensor([x * 255 for x in data_config['std']]).view(1, 3, 1, 1)

preprocess = create_transform(
    data_config['input_size'],
    is_training=False,
    use_prefetcher=True,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment=None,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    crop_pct=data_config['crop_pct'],
    tf_preprocessing=False,
    re_prob=0,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    separate=False,
)


# def gen_input_bin(file_batches, batch, src_path, save_path, model):
def gen_input_bin(file_batches, batch, src_path, save_path, model):
    """generate input bin file

    Args:
        file_batches (list): batch of filenames
        batch (int): batch index
        src_path (str): input image folder
        save_path (str): folder to save bin files

    Raises:
        ValueError: Invalid image name
    """
    i = 0
    for filename in file_batches[batch]:
        if ".db" in filename:
            continue
        i = i + 1
        print("batch", batch, filename, "===", i)
        if filename.endswith('.JPEG'):
            imgname = filename.strip('.JPEG')
        elif filename.endswith('.jpeg'):
            imgname = filename.strip('.jpeg')
        else:
            raise ValueError('Invalid image name:', filename)
        input_image = Image.open(os.path.join(
            src_path, filename)).convert('RGB')
        if '/' in imgname:
            _, imgname = imgname.split('/')
        input_tensor = preprocess(input_image)
        input_tensor = torch.tensor(input_tensor).unsqueeze(0)
        input_tensor = input_tensor.half().sub_(mean).div_(std)
        input_tensor = model.patch_embed(
            input_tensor.to(torch.float32)) + model.inner_pos
        input_tensor = input_tensor.unsqueeze(0)
        img = np.array(input_tensor.detach()).astype(np.float32)
        img.tofile(os.path.join(save_path, imgname + ".bin"))


def TNT_preprocess(src_path, save_path, model):
    """preprocess

    Args:
        src_path (str): folder of input images
        save_path (str): folder to save bin files
    """
    folder_list = os.listdir(src_path)
    if folder_list[0].endswith('.JPEG'):
        # val/xxxx.JPEG
        files = folder_list
    else:
        # val/xxxx/xxxx.JPEG
        files = []
        for folder in folder_list:
            file_list = os.listdir(os.path.join(src_path, folder))
            for filename in file_list:
                files.append(os.path.join(folder, filename))
    file_batches = [files]
    gen_input_bin(file_batches, 0, src_path, save_path, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess script for TNT \
        model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src-path', default='',
                        type=str, help='path of imagenet')
    parser.add_argument('--save-path', default='', type=str,
                        help='path to save bin files')
    args = parser.parse_args()
    args.distributed = False

    model = create_model(
        'tnt_s_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_connect_rate=None,  # DEPRECATED, use drop_path
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        checkpoint_path='')
    state_dict = torch.load('./tnt_s_81.5.pth.tar', map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if not os.path.isdir(args.save_path):
        os.makedirs(os.path.realpath(args.save_path))
    TNT_preprocess(args.src_path, args.save_path, model)
