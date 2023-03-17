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
import json
import argparse
import numpy as np
import cv2

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./coco2017.info")
    parser.add_argument("--det_results_path", default="./detection-results")
    parser.add_argument("--img_path", default="coco/val2017/", help="required when ifShowDetObj is open")
    parser.add_argument("--net_out_num", default=2)
    parser.add_argument("--net_input_width", default=640)
    parser.add_argument("--net_input_height", default=640)
    parser.add_argument("--prob_thres", default=0.05)
    parser.add_argument("--ifShowDetObj", action="store_true", help="show boxes in the pictures.")
    flags = parser.parse_args()
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
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name[:name.rfind('_')]
                     for name in os.listdir(bin_path) if "bin" in name])

    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        # load all detected output tensor
        res_buff = []
        for num in range(flags.net_out_num):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num==0:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 5])
                elif num==1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int64")
                    buf = np.reshape(buf, [100, 1])
                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")
        res_tensor = np.concatenate(res_buff, axis=1)
        current_img_size = img_size_dict[bin_file]
        predbox = coco_postprocess(res_tensor, current_img_size, flags.net_input_width, flags.net_input_height)

        if flags.ifShowDetObj == True:
            img_path = flags.img_path
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

        if flags.ifShowDetObj == True:
            print(os.path.join(det_results_path, bin_file +'.jpg'))
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY),70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)