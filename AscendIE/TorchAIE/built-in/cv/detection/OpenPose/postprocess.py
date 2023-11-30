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
import argparse
import json
import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

sys.path.append("./lightweight-human-pose-estimation.pytorch")
from modules.keypoints import group_keypoints, extract_keypoints
from val import run_coco_eval, convert_to_coco_format


def read_txt(txt_path, shape):
    with open(txt_path, "r") as f:
        line = f.read().splitlines()
        line_split = "".join(line).strip().split(" ")
        line_split = [eval(i) for i in line_split]
        line_split = torch.Tensor(line_split)
        heatmaps = line_split.view(shape)
    return heatmaps


def transfer(heatmaps, pafs, height, width, top, bottom, left, right, stride=8):
    heatmaps = np.transpose(heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(
        heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC
    )
    heatmaps = heatmaps[
        top : heatmaps.shape[0] - bottom, left : heatmaps.shape[1] - right :, :
    ]
    heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
    pafs = np.transpose(pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    pafs = pafs[top : pafs.shape[0] - bottom, left : pafs.shape[1] - right, :]
    pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
    return heatmaps, pafs


def post_process(args):
    result_folder = args.result_folder
    json_path = args.detections_save_path
    pad_path = args.pad_txt_path
    pad_info = {}
    with open(pad_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.strip().split(" ")
            pad_info[line_split[0]] = [int(line_split[i]) for i in range(1, 7)]
    pcm_files, paf_files = [], []
    for result_file in os.listdir(result_folder):
        img_name = result_file.split(".")[0]
        index = img_name.rfind("_")
        name_suffix = img_name[index + 1]
        if name_suffix == "0":
            pcm_files.append(result_file)
        elif name_suffix == "1":
            paf_files.append(result_file)
    pcm_files.sort()
    paf_files.sort()
    coco_result = []
    for pcm_file, paf_file in tqdm(tzip(pcm_files, paf_files)):
        img_name = pcm_file.split(".")[0]
        index = img_name.rfind("_")
        img_name = img_name[0:index] + ".jpg"
        pcm_file_path = os.path.join(result_folder, pcm_file)
        paf_file_path = os.path.join(result_folder, paf_file)

        heatmaps = torch.from_numpy(np.fromfile(pcm_file_path, np.float32)).view(
            1, 19, 46, 80
        )
        pafs = torch.from_numpy(np.fromfile(paf_file_path, np.float32)).view(
            1, 38, 46, 80
        )

        pad = pad_info[img_name]
        height, width = pad[0], pad[1]
        top, bottom, left, right = pad[2], pad[3], pad[4], pad[5]
        heatmaps, pafs = transfer(
            heatmaps, pafs, height, width, top, bottom, left, right
        )
        all_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            all_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, all_keypoints_num
            )
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
        image_id = int(img_name[0 : img_name.rfind(".")])
        for idx in range(len(coco_keypoints)):
            coco_result.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": coco_keypoints[idx],
                    "score": scores[idx],
                }
            )
    with open(json_path, "w") as f:
        json.dump(coco_result, f, indent=4)
    run_coco_eval(args.labels, json_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-folder", default="output/aie_inference_result")
    parser.add_argument(
        "--detections-save-path", default="./output/aie_inference_result.json"
    )
    parser.add_argument(
        "--pad-txt-path",
        default="./output/pad.txt",
        help="padding around the image with 368*640",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="/data/datasets/coco/annotations/person_keypoints_val2017.json",
        help="path to json with keypoints val labels",
    )
    args = parser.parse_args()

    post_process(args)


if __name__ == "__main__":
    main()
