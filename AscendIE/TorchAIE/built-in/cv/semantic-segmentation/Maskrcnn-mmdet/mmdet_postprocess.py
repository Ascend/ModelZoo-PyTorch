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

import os
import sys
import argparse
import cv2
import numpy as np
import pickle
import glob
sys.path.append("./mmdetection")

import torch
from torchvision.models.detection.roi_heads import paste_masks_in_image
import torch.nn.functional as F

from mmdetection.mmdet.core import bbox2result
from mmdetection.mmdet.core import encode_mask_results
from mmdetection.mmdet.datasets import CocoDataset


def postprocess_bboxes(bboxes, image_size, net_input_width, net_input_height):
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pad_w = net_input_width - w * scale
    pad_h = net_input_height - h * scale
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    bboxes[:, 0] = (bboxes[:, 0] - pad_left) / scale
    bboxes[:, 1] = (bboxes[:, 1] - pad_top)  / scale
    bboxes[:, 2] = (bboxes[:, 2] - pad_left) / scale
    bboxes[:, 3] = (bboxes[:, 3] - pad_top)  / scale

    return bboxes


def postprocess_masks(masks, image_size, net_input_width, net_input_height):
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pad_w = net_input_width - w * scale
    pad_h = net_input_height - h * scale
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    if pad_top < 0:
        pad_top = 0
    if pad_left < 0:
        pad_left = 0
    top = int(pad_top)
    left = int(pad_left)
    hs = int(pad_top + net_input_height - pad_h)
    ws = int(pad_left + net_input_width - pad_w)
    masks = masks.to(dtype=torch.float32)
    res_append = torch.zeros(0, h, w)
    if torch.cuda.is_available():
        res_append = res_append.to(device='cuda')
    for i in range(masks.size(0)):
        mask = masks[i][0][top:hs, left:ws]
        mask = mask.expand((1, 1, mask.size(0), mask.size(1)))
        mask = F.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
        mask = mask[0][0]
        mask = mask.unsqueeze(0)
        res_append = torch.cat((res_append, mask))

    return res_append[:, None]


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result")
    parser.add_argument("--test_annotation", default="./val2017.info")
    parser.add_argument("--det_results_path", default="./precision_result")
    parser.add_argument("--val2017_json_path", default="./coco/annotations/instances_val2017.json")
    parser.add_argument("--net_out_num", type=int, default=3)
    parser.add_argument("--net_input_width", type=int, default=1216)
    parser.add_argument("--net_input_height", type=int, default=1216)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    flags = parser.parse_args()

    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    bin_path = flags.bin_data_path
    det_results_path = flags.det_results_path
    os.makedirs(det_results_path, exist_ok=True)

    coco_dataset = CocoDataset(ann_file=flags.val2017_json_path, pipeline=[])
    coco_class_map = {id:name for id, name in enumerate(coco_dataset.CLASSES)}
    results = []

    cnt = 0
    for ids in coco_dataset.img_ids:
        cnt = cnt + 1

        bin_file = glob.glob(bin_path + '/*0' + str(ids) + '_0.bin')[0]
        bin_file = bin_file[bin_file.rfind('/') + 1:]
        bin_file = bin_file[:bin_file.rfind('_')]

        print(cnt - 1, bin_file)
        path_base = os.path.join(bin_path, bin_file)
        res_buff = []
        bbox_results = []
        cls_segms = []
        for num in range(0, flags.net_out_num):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 0:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 5])
                elif num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int32")
                    buf = np.reshape(buf, [100, 1])
                elif num == 2:
                    bboxes = np.fromfile(path_base + "_" + str(num - 2) + ".bin", dtype="float32")
                    bboxes = np.reshape(bboxes, [100, 5])
                    bboxes = torch.from_numpy(bboxes)
                    labels = np.fromfile(path_base + "_" + str(num - 1) + ".bin", dtype="int32")
                    labels = np.reshape(labels, [100, 1]).astype(np.int64)
                    labels = torch.from_numpy(labels)
                    mask_pred = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    mask_pred = np.reshape(mask_pred, [100, 80, 28, 28])
                    mask_pred = torch.from_numpy(mask_pred)

                    if torch.cuda.is_available():
                        mask_pred = mask_pred.to(device='cuda')

                    img_shape = (flags.net_input_height, flags.net_input_width)
                    mask_pred = mask_pred[range(len(mask_pred)), labels[:, 0]][:, None]
                    masks = paste_masks_in_image(mask_pred, bboxes[:, :4], img_shape)
                    masks = masks >= 0.5

                    
                    img_id = bin_file
                    masks = postprocess_masks(masks, img_size_dict[img_id], flags.net_input_width, flags.net_input_height)
                    if torch.cuda.is_available():
                        masks = masks.cpu()

                    cls_segms = [[] for _ in range(80)]
                    for i in range(len(masks)):
                        cls_segms[labels[i][0]].append(masks[i][0].numpy())

                    bboxes = postprocess_bboxes(bboxes, img_size_dict[img_id], flags.net_input_width, flags.net_input_height)
                    bbox_results = [bbox2result(bboxes, labels[:, 0], 80)]
                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")

        result = list(zip(bbox_results, [cls_segms]))
        result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
        results.extend(result)

        current_img_size = img_size_dict[img_id]
        res_bboxes = np.concatenate(res_buff, axis=1)
        predbox = postprocess_bboxes(res_bboxes, current_img_size, flags.net_input_width, flags.net_input_height)

        if flags.ifShowDetObj == True:
            imgCur = cv2.imread(current_img_size[2])

        det_results_str = ''
        for idx, class_idx in enumerate(predbox[:, 5]):
            if float(predbox[idx][4]) < float(0.05):
                continue
            if class_idx < 0 or class_idx > 80:
                continue

            class_name = coco_class_map[int(class_idx)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])
            if flags.ifShowDetObj == True:
                imgCur = cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 2)
                imgCur = cv2.putText(imgCur, class_name, (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if flags.ifShowDetObj == True:
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)

    save_variable(results, './results.txt')
    eval_results = coco_dataset.evaluate(results, metric=['bbox', 'segm'], classwise=True)
    