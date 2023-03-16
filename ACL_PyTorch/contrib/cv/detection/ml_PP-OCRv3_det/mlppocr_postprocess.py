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
import pickle
from pathlib import Path

import cv2
import paddle
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath('./PaddleOCR'))
from ppocr.postprocess import build_post_process
import tools.program as program

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)


def postprocess(config, info_path, res_dir, img_dir, save_dir):
    res_dir = Path(res_dir)
    img_dir = Path(img_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    post_process_class = build_post_process(config['PostProcess'])
    with open(info_path, 'rb') as f:
        info = pickle.load(f)
    for res_file in tqdm(res_dir.iterdir()):
        om_preds = dict(Student=dict(maps=np.load(str(res_file))))
        file_name = res_file.name.replace('_0.npy', '')
        shape_list = info[f'{file_name}.jpg']
        res = post_process_class(om_preds, shape_list)
        boxes = res['Student'][0]['points']
        
        img_file = img_dir / f'{file_name}.jpg'
        src_img = cv2.imread(str(img_file))
        draw_det_res(boxes, config, src_img, str(img_file), str(save_dir))


def main():
    config, _, logger, _ = program.preprocess()
    info_path = config['info_path']
    res_dir = config['res_dir']
    img_dir = config['Global']['infer_img']
    save_dir = config['save_dir']
    postprocess(config, info_path, res_dir, img_dir, save_dir)


if __name__ == '__main__':
    main()
