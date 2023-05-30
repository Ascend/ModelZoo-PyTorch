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
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def get_jpg_info(file_path):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_names = []
    res=[]
    for extension in extensions:
        image_names.append(glob(os.path.join(file_path, '*.' + extension)))
    for image_name in image_names:
        if len(image_name) == 0:
            continue
        else:
            for index, img in enumerate(image_name):
                img_cv = cv2.imread(img)
                shape = img_cv.shape
                width, height = shape[1], shape[0]
                content = [img, str(width), str(height)]
        res.append(content)
    return res

def coco_postprocess(bbox: np.ndarray, image_size, 
                        net_input_width, net_input_height):
    """
    This function is postprocessing for Retinanet output.

    Before calling this function, reshape the raw output to
    following form
        numpy.ndarray:
            [x, y, width, height, confidence, probability of 80 classes]
        shape: (100,)
    The postprocessing restore the bounding rectangles of Retinanet output
    to origin scale and filter with non-maximum suppression.

    :param bbox: a numpy array of the Retinanet output
    :param image_path: a string of image path
    :return: three list for best bound, class and score
    """
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pad_w = net_input_width - w * scale
    pad_h = net_input_height - h * scale
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # cal predict box on the image src
    pbox = bbox
    pbox[:, 0] = (bbox[:, 0] - pad_left) / scale
    pbox[:, 1] = (bbox[:, 1] - pad_top)  / scale
    pbox[:, 2] = (bbox[:, 2] - pad_left) / scale
    pbox[:, 3] = (bbox[:, 3] - pad_top)  / scale
    return pbox

    
def coco_evaluation(annotation_json, result_json):
    cocoGt = COCO(annotation_json)
    cocoDt = cocoGt.loadRes(result_json)
    iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    iou_type = 'bbox'

    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.params.catIds = cocoGt.getCatIds(catNms=CLASSES)
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.params.maxDets = [100, 300, 1000] # proposal number for evaluating recalls/mAPs.
    cocoEval.params.iouThrs = iou_thrs

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    val = float(f'{cocoEval.stats[0]:.3f}')
    print('mAP:', val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./output_data")
    parser.add_argument("--test_annotation", default="./data/coco/val2017")
    parser.add_argument("--ground_truth", default="instances_val2017.json")
    parser.add_argument("--json_output_file", default="coco_detection_result.json")
    parser.add_argument("--net_out_num", default=2)
    parser.add_argument("--net_input_width", default=1216)
    parser.add_argument("--net_input_height", default=1216)
    parser.add_argument("--prob_thres", default=0.05)
    flags = parser.parse_args()
    # generate dict according to annotation file for query resolution
    # load width and height of input images
    img_size_dict = dict()
    jpg_info = get_jpg_info(flags.test_annotation)
    for temp in jpg_info:
        img_file_path = temp[0]
        img_name = temp[0].split("/")[-1].split(".")[0]
        img_width = int(temp[1])
        img_height = int(temp[2])
        img_size_dict[img_name] = (img_width, img_height, img_file_path)

    # read bin file for generate predict result
    bin_path = flags.bin_data_path
    total_img = set([name[:name.rfind('_')]
                     for name in os.listdir(bin_path) if "bin" in name])
    bounding_boxes = []
    for bin_file in tqdm(sorted(total_img)):
        path_base = os.path.join(bin_path, bin_file)
        # load all detected output tensor
        res_buff = []
        buf1 = []
        buf2 = []
        for num in range(flags.net_out_num):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 0:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype=np.float32)
                    if buf.shape == (500,):
                        buf1 = np.reshape(buf, [100, 5])
                    else:
                        buf.dtype = np.int64
                        buf2 = np.reshape(buf,[100,1])
                elif num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype=np.int64)
                    if buf.shape == (100,): 
                        buf2 = np.reshape(buf, [100, 1])
                    else:
                        buf.dtype = np.float32
                        buf1 = np.reshape(buf, [100,5])
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")
        res_buff.append(buf1)
        res_buff.append(buf2)
        res_tensor = np.concatenate(res_buff, axis=1)
        current_img_size = img_size_dict[bin_file]
        predbox = coco_postprocess(res_tensor, current_img_size, flags.net_input_width, flags.net_input_height)

        for idx, class_ind in enumerate(predbox[:,5]):
            if float(predbox[idx][4]) < float(flags.prob_thres):
                continue
            # skip negative class index
            if class_ind < 0 or class_ind > 80:
                continue

            class_name = CLASSES[int(class_ind)]
            scores = predbox[idx][4]
            left = predbox[idx][0]
            top = predbox[idx][1]
            right = predbox[idx][2]
            bottom = predbox[idx][3]
            bbox_cur = [left, top, right-left, bottom-top]
            bounding_boxes.append({"image_id" : int(bin_file), "bbox" : bbox_cur, 
                                   "score" : float(scores), "category_id" : cat_ids[CLASSES.index(class_name)]})

    mmcv.dump(bounding_boxes, flags.json_output_file)
    coco_evaluation(flags.ground_truth, flags.json_output_file)
