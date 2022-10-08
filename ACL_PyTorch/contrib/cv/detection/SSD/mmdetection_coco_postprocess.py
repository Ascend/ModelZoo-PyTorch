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
"""coco postprocess"""

import os
import numpy as np
import argparse
import cv2
import warnings
import torch
import time
try:
    from torch import npu_batch_nms as NMSOp
    NMS_ON_NPU = True
except:
    from torchvision.ops import batched_nms as NMSOp
    NMS_ON_NPU = False

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


def coco_postprocess(bbox, image_size, net_input_width, net_input_height):
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
    scale_w = net_input_width / w
    scale_h = net_input_height / h

    # cal predict box on the image src
    pbox = bbox.copy()
    pbox[:, 0] = (bbox[:, 0]) / scale_w
    pbox[:, 1] = (bbox[:, 1]) / scale_h
    pbox[:, 2] = (bbox[:, 2]) / scale_w
    pbox[:, 3] = (bbox[:, 3]) / scale_h
    return pbox


def np_clip_bbox(bboxes, max_shape):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    h, w = max_shape
    x1 = x1.clip(min=0, max=w)
    y1 = y1.clip(min=0, max=h)
    x2 = x2.clip(min=0, max=w)
    y2 = y2.clip(min=0, max=h)
    bboxes = np.stack([x1, y1, x2, y2], axis=-1)
    return bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./coco2017_jpg.info")
    parser.add_argument("--det_results_path", default="./detection-results/")
    parser.add_argument("--net_out_num", default=2, type=int)
    parser.add_argument("--num_pred_box", default=8732, type=int)
    parser.add_argument("--nms_pre", default=-1, type=int)
    parser.add_argument("--net_input_width", default=300, type=int)
    parser.add_argument("--net_input_height", default=300, type=int)
    parser.add_argument("--min_bbox_size", default=0.01, type=float)
    parser.add_argument("--score_threshold", default=0.02, type=float)
    parser.add_argument("--nms", default=True, type=bool)
    parser.add_argument("--iou_threshold", default=0.45, type=float)
    parser.add_argument("--max_per_img", default=200, type=int)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    parser.add_argument("--start", default=0, type=float)
    parser.add_argument("--end", default=1, type=float)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--clear_cache", action='store_true')
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
    total_img = set([name[:name.rfind('_')] for name in os.listdir(bin_path) if "bin" in name])
    total_img = sorted(total_img)
    num_img = len(total_img)
    start = int(flags.start * num_img)
    end = int(flags.end * num_img)
    task_len = end - start + 1

    finished = 0
    time_start = time.time()
    for img_id in range(start, end):
    # for img_id, bin_file in enumerate(sorted(total_img)):
        bin_file = total_img[img_id]
        path_base = os.path.join(bin_path, bin_file)
        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        if os.path.exists(det_results_file) and not flags.clear_cache:
            continue

        # load all detected output tensor
        bbox_file = path_base + "_" + str(0) + ".bin"
        score_file = path_base + "_" + str(1) + ".bin"
        assert os.path.exists(bbox_file), '[ERROR] file `{}` not exist'.format(bbox_file)
        assert os.path.exists(score_file), '[ERROR] file `{}` not exist'.format(score_file)
        bboxes = np.fromfile(bbox_file, dtype="float32").reshape(flags.num_pred_box, 4)
        scores = np.fromfile(score_file, dtype="float32").reshape(flags.num_pred_box, 80)

        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        try:
            bboxes = bboxes.npu(flags.device)
            scores = scores.npu(flags.device)
        except:
            warnings.warn('npu is not available, running on cpu')
        
        max_scores, _ = scores.max(-1)
        keep_inds = (max_scores > flags.score_threshold).nonzero(as_tuple=False).view(-1)
        bboxes = bboxes[keep_inds, :]
        scores = scores[keep_inds, :]

        if flags.nms_pre > 0 and flags.nms_pre < bboxes.shape[0]:
            max_scores, _ = scores.max(-1)
            _, topk_inds = max_scores.topk(flags.nms_pre)
            bboxes = bboxes[topk_inds, :]
            scores = scores[topk_inds, :]
        
        # clip bbox border
        bboxes[:, 0::2].clamp_(min=0, max=flags.net_input_width - 1)
        bboxes[:, 1::2].clamp_(min=0, max=flags.net_input_height - 1)

        # remove small bbox
        bboxes_width_height = bboxes[:, 2:] - bboxes[:, :2]
        valid_bboxes = bboxes_width_height > flags.min_bbox_size
        keep_inds = (valid_bboxes[:, 0] & valid_bboxes[:, 1]
            ).nonzero(as_tuple=False).view(-1)
        bboxes = bboxes[keep_inds, :]
        scores = scores[keep_inds, :]

        # rescale bbox to original image size
        original_img_info = img_size_dict[bin_file]
        rescale_factor = torch.tensor([
            original_img_info[0] / flags.net_input_width,
            original_img_info[1] / flags.net_input_height] * 2,
            dtype=bboxes.dtype, device=bboxes.device)
        bboxes *= rescale_factor

        if flags.nms:
            if NMS_ON_NPU:
                # repeat bbox for each class
                # (N, 4) -> (B, N, 80, 4), where B = 1 is the batchsize
                bboxes = bboxes[None, :, None, :].repeat(1, 1, 80, 1)
                # (N, 80) -> (B, N, 80), where B = 1 is the batchsize
                scores = scores[None, :, :]

                # bbox batched nms
                bboxes, scores, labels, num_total_bboxes = \
                    NMSOp(
                        bboxes.half(), scores.half(),
                        score_threshold=flags.score_threshold,
                        iou_threshold=flags.iou_threshold,
                        max_size_per_class=flags.max_per_img,
                        max_total_size=flags.max_per_img)
                bboxes = bboxes[0, :num_total_bboxes, :]
                scores = scores[0, :num_total_bboxes]
                class_idxs = labels[0, :num_total_bboxes]
            else:
                # repeat bbox and class idx for each class
                bboxes = bboxes[:, None, :].repeat(1, 80, 1) # (N, 4) -> (N, 80, 4)
                class_idxs = torch.arange(80, dtype=torch.long, device=bboxes.device
                    )[None, :].repeat(bboxes.shape[0], 1) # (80) -> (N, 80)

                # reshape bbox for torch nms
                bboxes = bboxes.view(-1, 4)
                scores = scores.view(-1)
                class_idxs = class_idxs.view(-1)

                # bbox batched nms
                keep_inds = NMSOp(bboxes, scores, class_idxs, flags.iou_threshold)
                bboxes = bboxes[keep_inds]
                scores = scores[keep_inds]
                class_idxs = class_idxs[keep_inds]
        else:
            # repeat bbox and class idx for each class
            bboxes = bboxes[:, None, :].repeat(1, 80, 1) # (N, 4) -> (N, 80, 4)
            class_idxs = torch.arange(80, dtype=torch.long, device=bboxes.device
                )[None, :].repeat(bboxes.shape[0], 1) # (80) -> (N, 80)
            
            # reshape bbox for torch nms
            bboxes = bboxes.view(-1, 4)
            scores = scores.view(-1)
            class_idxs = class_idxs.view(-1)

        # keep topk max_per_img bbox
        if flags.max_per_img > 0 and flags.max_per_img < bboxes.shape[0]:
            _, topk_inds = scores.topk(flags.max_per_img)
            bboxes = bboxes[topk_inds, :]
            scores = scores[topk_inds]
            class_idxs = class_idxs[topk_inds]

        # move to cpu if running on npu
        if bboxes.device != 'cpu':
            bboxes = bboxes.cpu()
            scores = scores.cpu()
            class_idxs = class_idxs.cpu()
        
        # convert to numpy.ndarray
        bboxes = bboxes.numpy()
        scores = scores.numpy()
        class_idxs = class_idxs.numpy()

        # make det result file
        if flags.ifShowDetObj == True:
            imgCur = cv2.imread(original_img_info[2])

        det_results_str = ''
        for idx in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[idx, :]
            predscore = scores[idx]
            class_ind = class_idxs[idx]

            class_name = CLASSES[int(class_ind)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, predscore, x1, y1, x2, y2)
            if flags.ifShowDetObj == True:
                imgCur=cv2.rectangle(imgCur, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                imgCur = cv2.putText(imgCur, class_name + '|' + str(predscore),
                                    (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (0, 0, 255), 1)
            
        if flags.ifShowDetObj == True:
            cv2.imwrite(os.path.join(det_results_path, bin_file + '.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)

        finished += 1
        speed = finished / (time.time() - time_start) 
        print('processed {:5d}/{:<5d} images, speed: {:.2f}FPS'.format(finished, task_len, speed), end='\r')
