# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
import argparse
import copy
import os
import cv2
import shutil
import math
import numpy as np
import torch
import torch.nn.functional as F
from ctpn import config
from ctpn.ctpn import CTPN_Model
from ctpn.utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch CTPN Testing')
parser.add_argument('--data-path', default='/home/dockerHome/ctpn/ctpn_8p/imagedata/', type=str,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--model-path', default='./output_models/checkpoint-200.pth.tar', type=str,
                    help='number of total epochs to run')
args = parser.parse_args()


def load_state_dict(state_dicts):
    new_state_dicts = {}
    for k, v in state_dicts.items():
        if (k[:7] == 'module.'):
            name = k[7:]
        else:
            name = k
        new_state_dicts[name] = v
    return new_state_dicts


long_len = 1008

device = 'npu:0'
weights = args.model_path
model = CTPN_Model().to(device)
state_dict = torch.load(weights, map_location='cpu')['state_dict']
new_state_dict = load_state_dict(state_dict)
model.load_state_dict(new_state_dict)
model.eval()


def get_text_boxes(image, img_name=None, display=True, prob_thresh=0.5):
    h, w = image.shape[:2]
    image_c = copy.deepcopy(image)
    image = image[:, :, ::-1]
    rescale_fac = max(h, w) / long_len
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w, h))
        h, w = image.shape[:2]
    target_w = int(math.ceil(w / 16)) * 16
    target_h = int(math.ceil(h / 16)) * 16

    w_pad = (target_w - w) // 2
    h_pad = (target_h - h) // 2

    w_pad2 = target_w - w - w_pad
    h_pad2 = target_h - h - h_pad
    image = np.pad(image, ((h_pad, h_pad2), (w_pad, w_pad2), (0, 0)))
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    h, w = target_h, target_w
    with torch.no_grad():
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = transform_bbox(anchor, regr)
        bbox = clip_bbox(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        text_n = copy.deepcopy(text)
        text[:, [0, 2, 4, 6]] = text[:, [0, 2, 4, 6]] - w_pad
        text[:, [1, 3, 5, 7]] = text[:, [1, 3, 5, 7]] - h_pad
        if rescale_fac > 1.0:
            text_ori = text * rescale_fac
        else:
            text_ori = text
        if display:
            for i in text_ori:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 255, 0), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 255, 0), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 255, 0), 2)
                cv2.putText(image_c, s, (i[0] + 13, i[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            cv2.imwrite(f'./display/{img_name}', image_c)

        return text_n, text_ori, image_c  # 返回文字坐标，原始文字坐标，原图像


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def predict_2_txt():
    img_name_list = os.listdir(os.path.join(args.data_path, 'Challenge2_Test_Task12_Images'))
    make_dir('./display/')
    make_dir('./results/')
    make_dir('./results/predict_txt_2/')
    for k in tqdm(range(len(img_name_list))):
        img_path = os.path.join(args.data_path, 'Challenge2_Test_Task12_Images/{}'.format(img_name_list[k]))
        input_img = cv2.imread(img_path)
        text, text_ori, out_img = get_text_boxes(input_img, img_name=img_name_list[k], display=True)
        with open('./results/predict_txt_2/res_{}.txt'.format(img_name_list[k][:-4]), 'w') as f:
            for i in range(text_ori.shape[0]):
                x_min, y_min = min(text_ori[i][0:8:2]), min(text_ori[i][1:8:2])
                x_max, y_max = max(text_ori[i][0:8:2]), max(text_ori[i][1:8:2])
                f.write('{},{},{},{}'.format(int(x_min), int(y_min), int(x_max), int(y_max)))
                f.write('\n')


if __name__ == '__main__':
    predict_2_txt()
