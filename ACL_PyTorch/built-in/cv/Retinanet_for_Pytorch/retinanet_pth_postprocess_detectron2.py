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
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import pickle
import logging

class CocoDataset:
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

def postprocess_bboxes(bboxes, image_size, net_input_width, net_input_height):
    old_h = image_size[0]
    old_w = image_size[1]
    scale_ratio = 800 / min(old_w, old_h)
    if old_h < old_w:
        new_h, new_w = 800, int(np.floor(scale_ratio * old_w))
    else:
        new_h, new_w = int(np.floor(scale_ratio * old_h)), 800
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)
    scale = new_w/old_w

    bboxes[:, 0] = (bboxes[:, 0]) / scale
    bboxes[:, 1] = (bboxes[:, 1]) / scale
    bboxes[:, 2] = (bboxes[:, 2]) / scale
    bboxes[:, 3] = (bboxes[:, 3]) / scale

    return bboxes


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
    parser.add_argument("--test_annotation", default="./origin_image.info")
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0/")
    parser.add_argument("--val2017_path", default="/root/datasets/coco/val2017/")
    parser.add_argument("--det_results_path", default="./result/detection-results/")
    parser.add_argument("--net_out_num", type=int, default=3)
    parser.add_argument("--net_input_width", type=int, default=1344)
    parser.add_argument("--net_input_height", type=int, default=1344)
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
    total_img = set([name[:name.rfind('_')] for name in os.listdir(bin_path) if "bin" in name])

    logging.basicConfig(level=logging.INFO)
    coco_path = flags.val2017_path
    dataloader_val = CocoDataset(coco_path, set_name='val2017')
    results = []
    image_ids = []

    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        res_buff = []
        for num in range(1, flags.net_out_num + 1):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    boxes = np.reshape(buf, [100, 4])
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int32")
                    labels = np.reshape(buf, [100, 1])
                elif num == 3:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    scores = np.reshape(buf, [100, 1])
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")

        current_img_size = img_size_dict[bin_file]
        boxes = postprocess_bboxes(boxes, current_img_size, flags.net_input_width, flags.net_input_height)

        if boxes.shape[0] > 0:
            # change to (x, y, w, h) (MS COCO standard)
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]
            for box_id in range(boxes.shape[0]):
                if scores[box_id] <0.05:
                    continue
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]
                image_result = {
                    'image_id': int(bin_file),
                    'category_id': dataloader_val.label_to_coco_label(label),
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                # append detection to results
                results.append(image_result)
            image_ids.append(int(bin_file))
    
    json.dump(results, open('{}_bbox_results.json'.format(dataloader_val.set_name), 'w'), indent=4)
    coco_true = dataloader_val.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataloader_val.set_name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()