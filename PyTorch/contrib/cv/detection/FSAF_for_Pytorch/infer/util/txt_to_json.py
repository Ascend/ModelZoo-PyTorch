# Copyright 2021 Huawei Technologies Co., Ltd
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
# limitations under the License.import argparse

import glob
import os
import sys
import argparse
import mmcv

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def error(msg):
    print(msg)
    sys.exit(0)


def get_predict_list(file_path, gt_classes):
    dr_files_list = glob.glob(file_path + '/*.txt')
    dr_files_list.sort()

    bounding_boxes = []
    for txt_file in dr_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                sl = line.split()
                if len(sl) > 6:
                    class_name = sl[0] + ' ' + sl[1]
                    scores, left, top, right, bottom = sl[2:]
                else:
                    class_name, scores, left, top, right, bottom = sl
                if float(scores) < 0.05:
                    continue
            except ValueError:
                error_msg = "Error: File " + txt_file + " wrong format.\n"
                error_msg += " Expected: <classname> <conf> <l> <t> <r> <b>\n"
                error_msg += " Received: " + line
                error(error_msg)

            # bbox = left + " " + top + " " + right + " " + bottom
            left = float(left)
            right = float(right)
            top = float(top)
            bottom = float(bottom)
            bbox = [left, top, right-left, bottom-top]
            bounding_boxes.append({"image_id": int(file_id), "bbox": bbox, 
                                   "score": float(scores), "category_id": cat_ids[CLASSES.index(class_name)]})
        # sort detection-results by decreasing scores
        # bounding_boxes.sort(key=lambda x: float(x['score']), reverse=True)
    return bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser('mAp calculate')
    parser.add_argument('--infer_result_path', default="../sdk/output/infer_result/",
                        help='the path of the predict result')
    parser.add_argument("--json_output_filename", default="../sdk/output/coco_detection_result")
    args = parser.parse_args()

    res_bbox = get_predict_list(args.infer_result_path, CLASSES)
    mmcv.dump(res_bbox, args.json_output_filename + '.json')