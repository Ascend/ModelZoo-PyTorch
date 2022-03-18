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

import sys
sys.path.append('./ctpn.pytorch')
import os
import argparse
import cv2
import numpy as np
from ctpn.utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented


def get_text_boxes_pth(image, img_name, args, prob_thresh=0.5):
    """[get pth model text boxes and generate coordinates]

    Args:
        image ([numpy]): [input image]
        img_name ([str]): [image name]
        args ([argparse]): [postprocess parameters]
        prob_thresh (float, optional): [prob thresh]. Defaults to 0.5.
    """
    h, w= image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w, h))
        h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

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
    
        if rescale_fac > 1.0:
            text_ori = text * rescale_fac
        else:
            text_ori = text

    with open(os.path.join(args.pth_txt, 'res_{}.txt'.format(img_name[:-4])), 'w') as f:
        for i in range(text_ori.shape[0]):
            x_min, y_min = min(text_ori[i][0:8:2]), min(text_ori[i][1:8:2])
            x_max, y_max = max(text_ori[i][0:8:2]), max(text_ori[i][1:8:2])
            f.write('{},{},{},{}'.format(int(x_min + 0.5), int(y_min + 0.5), int(x_max + 0.5), int(y_max + 0.5)))
            f.write('\n')


def get_text_boxes_om(image, img_name, args, prob_thresh=0.5):
    """[get om model text boxes and generate coordinates]

    Args:
        image ([numpy]): [input image]
        img_name ([str]): [image name]
        args ([argparse]): [postprocess parameters]
        prob_thresh (float, optional): [prob thresh]. Defaults to 0.5.
    """
    h, w= image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
    
    # Determine the corresponding width and height according to the minimum distance
    distance_list = []
    for n in range(config.center_len):
        distance_list.append(np.linalg.norm(np.array([h, w])-np.array(config.center_list[n])))
    min_distance = min(distance_list)
    side_h = config.center_list[distance_list.index(min_distance)][0]
    side_w = config.center_list[distance_list.index(min_distance)][1]

    # Read the output file of the om model
    cls_numpy = np.fromfile('./result/dumpOutput_device0/{}_1.bin'.format(img_name[:-4]), dtype="float32")
    cls_numpy = cls_numpy.reshape((-1, 2))[:int(side_h / 16)*int(side_w / 16)*10, :]
    regr_numpy = np.fromfile('./result/dumpOutput_device0/{}_2.bin'.format(img_name[:-4]), dtype="float32")
    regr = regr_numpy.reshape((1, -1, 2))[:, :int(side_h / 16)*int(side_w / 16)*10, :]
    
    cls_exp = np.exp(cls_numpy)
    cls_sum = np.sum(cls_exp, axis=1, keepdims=True)
    cls_ = cls_exp / cls_sum
    cls_prob = cls_.reshape((1, -1, 2))
    anchor = gen_anchor((int(side_h / 16), int(side_w / 16)), 16)
    bbox = transform_bbox(anchor, regr)
    bbox = clip_bbox(bbox, [side_h, side_w])

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
    text = textConn.get_text_lines(select_anchor, select_score, [side_h, side_w])
    
    with open(os.path.join(args.predict_txt, 'res_{}.txt'.format(img_name[:-4])), 'w') as f:
        for i in range(text.shape[0]):
            x_min, y_min = min(text[i][0:8:2]), min(text[i][1:8:2])
            x_max, y_max = max(text[i][0:8:2]), max(text[i][1:8:2])
            height, width = image.shape[:2]

            scale_h = side_h / height
            scale_w = side_w / width
            scale = min(scale_h, scale_w) # Calculation of scale with different width and height

            width_scaled = int(width * scale)
            height_scaled = int(height * scale)
            width_offset = (side_w - width_scaled) // 2
            height_offset = (side_h - height_scaled) // 2
            x_min, x_max = x_min - width_offset, x_max - width_offset
            y_min, y_max = y_min - height_offset, y_max - height_offset
            x_min, x_max = int(x_min / scale+0.5), int(x_max / scale+0.5)
            y_min, y_max = int(y_min / scale+0.5), int(y_max / scale+0.5)
            f.write('{},{},{},{}'.format(x_min, y_min, x_max, y_max))
            f.write('\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ctpn postprocess') # ctpn postprocess parameters
    parser.add_argument('--model', default='om',
                        type=str, help='om model or pth model')
    parser.add_argument('--imgs_dir', default='data/Challenge2_Test_Task12_Images',
                        type=str, help='images path')
    parser.add_argument('--bin_dir', default='result/dumpOutput_device0',
                        type=str, help='post bin path')
    parser.add_argument('--pth_txt', default='data/pth_txt',
                        type=str, help='pth predict txt path')
    parser.add_argument('--predict_txt', default='data/predict_txt',
                        type=str, help='predict txt path')            
    args = parser.parse_args()
    img_name_list = os.listdir(args.imgs_dir)
    if args.model == 'om':
        import config
        for k in range(len(img_name_list)):
            input_img_ori = cv2.imread(os.path.join(args.imgs_dir, '{}'.format(img_name_list[k]))) # input image original
            get_text_boxes_om(input_img_ori, img_name_list[k], args)
    else:
        import torch
        import torch.nn.functional as F
        from ctpn import config
        from ctpn.ctpn import CTPN_Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = './ctpn.pytorch/weights/ctpn.pth'
        model = CTPN_Model().to(device)
        model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
        model.eval()
        for k in range(len(img_name_list)):
            input_img_ori = cv2.imread(os.path.join(args.imgs_dir, '{}'.format(img_name_list[k])))
            get_text_boxes_pth(input_img_ori, img_name_list[k], args)
  