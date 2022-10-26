# Copyright 2020 Huawei Technologies Co., Ltd
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

import glob
import os
import sys
import argparse
import mmcv

name2id = {
	'person':1,
	'bicycle':2,
	'car':3,
	'motorcycle':4,
	'airplane':5,
	'bus':6,
	'train':7,
	'truck':8,
	'boat':9,
	'traffic light':10,
	'fire hydrant':11,
	'stop sign':13,
	'parking meter':14,
	'bench':15,
	'bird':16,
	'cat':17,
	'dog':18,
	'horse':19,
	'sheep':20,
	'cow':21,
	'elephant':22,
	'bear':23,
	'zebra':24,
	'giraffe':25,
	'backpack':27,
	'umbrella':28,
	'handbag':31,
	'tie':32,
	'suitcase':33,
	'frisbee':34,
	'skis':35,
	'snowboard':36,
	'sports ball':37,
	'kite':38,
	'baseball bat':39,
	'baseball glove':40,
	'skateboard':41,
	'surfboard':42,
	'tennis racket':43,
	'bottle':44,
	'wine glass':46,
	'cup':47,
	'fork':48,
	'knife':49,
	'spoon':50,
	'bowl':51,
	'banana':52,
	'apple':53,
	'sandwich':54,
	'orange':55,
	'broccoli':56,
	'carrot':57,
	'hot dog':58,
	'pizza':59,
	'donut':60,
	'cake':61,
	'chair':62,
	'couch':63,
	'potted plant':64,
	'bed':65,
	'dining table':67,
	'toilet':70,
	'tv':72,
	'laptop':73,
	'mouse':74,
	'remote':75,
	'keyboard':76,
	'cell phone':77,
	'microwave':78,
	'oven':79,
	'toaster':80,
	'sink':81,
	'refrigerator':82,
	'book':84,
	'clock':85,
	'vase':86,
	'scissors':87,
	'teddy bear':88,
	'hair drier':89,
	'toothbrush':90
}
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


def get_predict_list(file_path):
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
                                   "score": float(scores), "category_id": name2id[class_name]})
        # sort detection-results by decreasing scores
        # bounding_boxes.sort(key=lambda x: float(x['score']), reverse=True)
    return bounding_boxes



if __name__ == '__main__':
    parser = argparse.ArgumentParser('mAp calculate')
    parser.add_argument('--npu_txt_path', default="detection-results",
                        help='the path of the predict result')
    parser.add_argument("--json_output_file", default="coco_detection_result")
    args = parser.parse_args()

    res_bbox = get_predict_list(args.npu_txt_path)
    mmcv.dump(res_bbox, args.json_output_file + '.json')