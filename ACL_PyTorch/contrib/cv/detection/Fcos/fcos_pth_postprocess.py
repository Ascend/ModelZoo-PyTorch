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
import argparse
import cv2
import numpy as np

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

import pickle
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
    parser.add_argument("--test_annotation", default="./origin_pictures.info")
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0/")
    parser.add_argument("--det_results_path", default="./detection-results/")
    parser.add_argument("--net_out_num", type=int, default=3)
    parser.add_argument("--net_input_width", type=int, default=1344)
    parser.add_argument("--net_input_height", type=int, default=1344)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    parser.add_argument("--annotations_path", default="/opt/npu/coco/annotations")
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

    import glob
    import torch
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    import torch.nn.functional as F
    from mmdet.core import bbox2result
    from mmdet.core import encode_mask_results
    from mmdet.datasets import CocoDataset
    coco_dataset = CocoDataset(ann_file='{}/instances_val2017.json'.format(flags.annotations_path), pipeline=[])
    coco_class_map = {id:name for id, name in enumerate(coco_dataset.CLASSES)}
    results = []

    cnt = 0
    for ids in coco_dataset.img_ids:
        cnt = cnt + 1
        bin_file = glob.glob(bin_path + '/*0' + str(ids) + '_1.bin')[0]
        bin_file = bin_file[bin_file.rfind('/') + 1:]
        bin_file = bin_file[:bin_file.rfind('_')]
        # print(cnt - 1, bin_file)
        path_base = os.path.join(bin_path, bin_file)
        res_buff = []
        bbox_results = []
        cls_segms = []

        if os.path.exists(path_base + "_" + "1" + ".bin") and os.path.exists(path_base + "_" + "2" + ".bin"):
            bboxes = np.fromfile(path_base + "_" + str(flags.net_out_num - 1) + ".bin", dtype="float32")
            bboxes = np.reshape(bboxes, [100, 5])
            bboxes = torch.from_numpy(bboxes)
            labels = np.fromfile(path_base + "_" + str(flags.net_out_num - 2) + ".bin", dtype="int64")
            labels = np.reshape(labels, [100, 1])
            labels = torch.from_numpy(labels)

            img_shape = (flags.net_input_height, flags.net_input_width)

            bboxes = postprocess_bboxes(bboxes, img_size_dict[bin_file], flags.net_input_width, flags.net_input_height)
            bbox_results = [bbox2result(bboxes, labels[:, 0], 80)]
        else:
            print("[ERROR] file not exist", path_base + "_" + str(1) + ".bin",path_base + "_" + str(2) + ".bin")

        result = bbox_results
        results.extend(result)

    # save_variable(results, './results.txt')
    eval_results = coco_dataset.evaluate(results, metric=['bbox', ], classwise=True)