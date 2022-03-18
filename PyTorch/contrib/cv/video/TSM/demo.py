# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import argparse
from PIL import Image
from apex import amp

import torch
import numpy as np
from torchvision import transforms

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmaction.models import build_model


def get_data(arg):
    # The model is used to process videos with 8 frames extracted from a video
    # arg.data_root represents the dataset path
    # arg.test_num represents which video to be tested
    info_path = os.path.join(arg.data_root, 'ucf101/ucf101_val_split_1_rawframes.txt')

    with open(info_path) as f:
        line = f.readlines()[arg.test_num].split(' ')
        video_path = line[0]
        frame_num = int(line[1])
        class_num = int(line[2])

    anno_path = os.path.join(arg.data_root, 'ucf101/annotations/classInd.txt')
    with open(anno_path) as f:
        class_name = f.readlines()

    frame_path = os.path.join(arg.data_root, 'ucf101/rawframes', video_path)
    frame_list = sorted(os.listdir(frame_path))

    split_len = frame_num // 10
    idx_list = [split_len * (x + 1) for x in range(8)]

    imgs = []
    for idx in idx_list:
        img_path = os.path.join(frame_path, frame_list[idx])
        img = Image.open(img_path).convert('RGB')
        imgs.append(img)
    return imgs, class_num, class_name


def test(arg):
    # generate random input
    img_input, true_class, class_name = get_data(arg)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transfrom = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    input = [data_transfrom(data) for data in img_input]
    input = torch.stack(input).unsqueeze(0).npu()

    # load config
    config_path = 'config/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py'
    cfg = Config.fromfile(config_path)

    # set device
    device = torch.device('npu:{}'.format(cfg.DEVICE_ID))
    torch.npu.set_device(device)

    # build model
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    if cfg.AMP:
        model = amp.initialize(model.npu(), opt_level=cfg.OPT_LEVEL, loss_scale=cfg.LOSS_SCALE)
    load_checkpoint(model, './result/epoch_32.pth', map_location='cpu')

    model = model.npu()
    model.eval()
    with torch.no_grad():
        output = model(input, return_loss=False)

    output = torch.from_numpy(output).type(torch.float32)
    _, pred = output.topk(1, 1, True, True)

    pred_class = pred[0][0].item()
    true_name = class_name[true_class].split(' ')[1][:-1]
    pred_name = class_name[pred_class].split(' ')[1][:-1]
    print("Prediction: Class Number - {}, Class Name - {}".format(pred_class, pred_name))
    print("Ground Truth: Class Number - {}, Class Name - {}".format(true_class, true_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--data_root', type=str, default='/opt/npu', help='Dataset saving path')
    parser.add_argument('--test_num', type=int, default=0, help='Choose the certain video for testing, starting from 0')

    args = parser.parse_args()
    test(args)
