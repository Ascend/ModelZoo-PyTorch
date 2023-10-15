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

from __future__ import print_function

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./Pytorch_Retinaface")
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from data import cfg_mnet
from utils.nms.py_cpu_nms import py_cpu_nms


def post_process(result_list, info_list, save_path, threshold):
    bin_images = glob(os.path.join(result_list, r"*.bin"))
    bin_images.sort()
    assert len(bin_images) == 9678
    cnt = 0
    i = 0
    loc, conf, landms = None, None, None
    scale = torch.ones(
        4,
    ).fill_(1000)
    for img in tqdm(bin_images):
        buf = np.fromfile(img, dtype="float32")
        if "_0.bin" in img:
            landms = np.reshape(buf, [1, 41236, 10])
            cnt = cnt + 1
        if "_1.bin" in img:
            conf = np.reshape(buf, [1, 41236, 2])
            cnt = cnt + 1
        if "_2.bin" in img:
            loc = np.reshape(buf, [1, 41236, 4])
            cnt = cnt + 1
        if cnt == 3:
            loc = torch.Tensor(loc)
            conf = torch.Tensor(conf)
            landms = torch.Tensor(landms)
            info_image = glob(
                os.path.join(info_list, "*/" + os.path.basename(img)[:-6] + ".bin")
            )
            resize = np.fromfile(info_image[0], dtype=np.float32)
            img_name = info_image[0]
            assert len(info_image) == 1
            i = i + 1
            cnt = 0
            priorbox = PriorBox(cfg_mnet, image_size=(1000, 1000))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
            boxes = boxes * scale / resize
            boxes = boxes.numpy()
            scores = conf.squeeze(0).data.numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
            scale1 = torch.ones(
                10,
            ).fill_(1000)

            landms = landms * scale1 / resize
            landms = landms.numpy()

            inds = np.where(scores > threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            # keep top-K before NMS
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False
            )
            keep = py_cpu_nms(dets, 0.4)
            dets = dets[keep, :]
            landms = landms[keep]

            dets = np.concatenate((dets, landms), axis=1)
            save_name = os.path.join(
                save_path,
                os.path.dirname(img_name).split("/")[-1],
                os.path.basename(img_name)[:-4] + ".txt",
            )
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = (
                        str(x)
                        + " "
                        + str(y)
                        + " "
                        + str(w)
                        + " "
                        + str(h)
                        + " "
                        + confidence
                        + " \n"
                    )
                    fd.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retinaface")
    parser.add_argument(
        "--prediction-folder",
        default="./result",
        type=str,
        help="infer prediction result path",
    )
    parser.add_argument(
        "--info-folder",
        default="./widerface/prep_info",
        type=str,
        help="input info path",
    )
    parser.add_argument(
        "--output-folder",
        default="./widerface_result/",
        type=str,
        help="Dir to save txt results",
    )
    parser.add_argument(
        "--confidence-threshold", default=0.02, type=float, help="confidence threshold"
    )
    args = parser.parse_args()
    post_process(
        args.prediction_folder,
        args.info_folder,
        args.output_folder,
        args.confidence_threshold,
    )
