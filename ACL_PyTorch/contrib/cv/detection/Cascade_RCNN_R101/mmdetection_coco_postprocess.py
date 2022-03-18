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

import os
import numpy as np
import cv2
import glob
import sys
import argparse
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

def coco_postprocess(bbox: np.ndarray, image_size, 
                        net_input_width, net_input_height):
    """
    This function is postprocessing for FasterRCNN output.

    Before calling this function, reshape the raw output of FasterRCNN to
    following form
        numpy.ndarray:
            [x, y, width, height, confidence, probability of 80 classes]
        shape: (100,)
    The postprocessing restore the bounding rectangles of FasterRCNN output
    to origin scale and filter with non-maximum suppression.

    :param bbox: a numpy array of the FasterRCNN output
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

    # make pbboxes value in valid range
    pbox[:, 0] = np.maximum(pbox[:, 0], 0)
    pbox[:, 1] = np.maximum(pbox[:, 1], 0)
    pbox[:, 2] = np.minimum(pbox[:, 2], w)
    pbox[:, 3] = np.minimum(pbox[:, 3], h)
    return pbox
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
def coco_evaluation(annotation_json, result_json):
    cocoGt = COCO(annotation_json)
    cocoDt = cocoGt.loadRes(result_json)
    iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    iou_type = 'bbox'

    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.params.catIds = cocoGt.get_cat_ids(cat_names=CLASSES)
    cocoEval.params.imgIds = cocoGt.get_img_ids()
    cocoEval.params.maxDets = [100, 300, 1000] # proposal number for evaluating recalls/mAPs.
    cocoEval.params.iouThrs = iou_thrs

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # mapping of cocoEval.stats
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    eval_results = {}

    for metric_item in metric_items:
        key = f'bbox_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results['bbox_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        f'{ap[4]:.3f} {ap[5]:.3f}')
    
    return eval_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./coco2017_jpg.info")
    parser.add_argument("--det_results_path", default="./detection-results")
    parser.add_argument("--img_path", default="./val2017/")
    parser.add_argument("--net_out_num", default=2)
    parser.add_argument("--net_input_width", default=1216)
    parser.add_argument("--net_input_height", default=1216)
    parser.add_argument("--prob_thres", default=0.05)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    parser.add_argument('--npu_txt_path', default="detection-results",
                        help='the path of the predict result')
    parser.add_argument("--json_output_file", default="coco_detection_result")
    parser.add_argument("--ground_truth", default="instances_val2017.json")
    parser.add_argument("--detection_result", default="coco_detection_result.json")
    flags = parser.parse_args()
    print(flags.ifShowDetObj, type(flags.ifShowDetObj))
    # generate dict according to annotation file for query resolution
    # load width and height of input images
    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    # read bin file for generate predict result
    bin_path = flags.bin_data_path
    det_results_path = flags.det_results_path
    img_path = flags.img_path
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name[:name.rfind('_')]
                     for name in os.listdir(bin_path) if "bin" in name])
    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        # load all detected output tensor
        res_buff = []
        for num in range(1, flags.net_out_num + 1):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 5])
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int64")
                    buf = np.reshape(buf, [100, 1])
                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")
        res_tensor = np.concatenate(res_buff, axis=1)
        current_img_size = img_size_dict[bin_file]
        print("[TEST]---------------------------concat{} imgsize{}".format(len(res_tensor), current_img_size))
        predbox = coco_postprocess(res_tensor, current_img_size, flags.net_input_width, flags.net_input_height)

        if flags.ifShowDetObj == True:
            pic = os.path.join(img_path, bin_file +'.jpg')
            imgCur = cv2.imread(pic)

        det_results_str = ''
        for idx, class_ind in enumerate(predbox[:,5]):
            if float(predbox[idx][4]) < float(flags.prob_thres):
                continue
            # skip negative class index
            if class_ind < 0 or class_ind > 80:
                continue

            class_name = CLASSES[int(class_ind)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])
            if flags.ifShowDetObj == True:
                imgCur=cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), 
                                    (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 1)
                imgCur = cv2.putText(imgCur, class_name+'|'+str(predbox[idx][4]), 
                                    (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                  # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
            
        if flags.ifShowDetObj == True:
            print(os.path.join(det_results_path, bin_file +'.jpg'))
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY),70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)
        print(det_results_str)

    res_bbox = get_predict_list(flags.npu_txt_path, CLASSES)
    mmcv.dump(res_bbox, flags.json_output_file + '.json')
    result = coco_evaluation(flags.ground_truth, flags.detection_result)
    print(result)
    with open('./coco_detection_result.txt', 'w') as f:
        for key, value in result.items():
            f.write(key + ': ' + str(value) + '\n')