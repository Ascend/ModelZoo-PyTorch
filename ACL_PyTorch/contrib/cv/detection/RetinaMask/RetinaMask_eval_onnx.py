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
# limitations under the License

import os
import sys
import time
import argparse
import numpy as np
import onnxruntime as ort
import pycocotools.coco as coco
import pycocotools.mask as mask_util

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

from tqdm import tqdm
from PIL import Image
from tools.utils import build_transforms, np_batched_nms, Masker, post_process, \
    convert_to_coco_format, evaluate_prediction

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str,
                        default="../weights/npu_8P_model_0020001_bs1_sim.onnx")
    parser.add_argument("--coco_path", type=str,
                        default=r"/opt/npu/coco")

    args = parser.parse_args()

    fix_shape = 1344
    trans = build_transforms(fix_shape)
    # load model
    onnx_session = ort.InferenceSession(args.weight_path, providers=['TensorrtExecutionProvider',
                                                                     'CUDAExecutionProvider'])
    masker = Masker(threshold=0.5, padding=1)

    # load images
    coco_dataset = coco.COCO(os.path.join(args.coco_path, 'annotations', 'instances_val2017.json'))
    imgs_path = os.path.join(args.coco_path, 'val2017')
    image_files = []
    for main_dir, sub_dir, files in os.walk(imgs_path):
        for file in files:
            if file.split('.')[-1] in ['jpg', 'png', 'bmp']:
                image_files.append(os.path.join(main_dir, file))
    image_files.sort()

    # run
    cost_time = []
    data_list = []
    for image_file in tqdm(image_files):
        image_file = image_file.replace('\\', '/')
        image_id = int(image_file.split('/')[-1].split('.')[0])

        img = Image.open(image_file).convert('RGB')
        ori_w, ori_h = img.size
        long_side = ori_w if ori_w > ori_h else ori_h
        ratio = fix_shape / long_side
        dummy_input = trans(img)

        # onnx process
        onnx_inputs = {onnx_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
        start_time = time.time()
        onnx_output = onnx_session.run(None, onnx_inputs)
        end_time = time.time()

        cost_time.append(end_time - start_time)
        detections = onnx_output[:3]
        masks = onnx_output[-1]

        # nms
        bboxes, scores, labels = detections[0], detections[2], detections[1]
        bboxes, labels, scores, masks = np_batched_nms(bboxes, scores, labels, masks, iou_threshold=0.4)

        # post
        bboxes, labels, scores, masks = post_process(ori_h, ori_w, ratio, bboxes, labels, scores, masks)
        masks = masker(masks, bboxes, ori_h, ori_w)

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F").astype('uint8'))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        data_list.extend(convert_to_coco_format(coco_dataset, image_id, bboxes, labels, scores, rles))

    mean_cost_time = np.array(cost_time).mean()
    fps = 1 / mean_cost_time
    print('FPS: %.4f ' % fps)

    bbox_info, segm_info = evaluate_prediction(coco_dataset, data_list)

    print('bbox_info: \n', bbox_info)
    print('segm_info: \n', segm_info)
