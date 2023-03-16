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
import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from utils.datasets import create_dataloader
from utils.general import non_max_suppression, scale_coords, colorstr, xyxy2xywh, coco80_to_coco91_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser('YoloV7 postprocess')
    parser.add_argument('--result_path', type=str, required=True, help='result files path.')
    parser.add_argument('--img_path', type=str, required=True, help='img dir.')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    args = parser.parse_args()
    if isinstance(args.data, str):
        is_coco = args.data.endswith('coco.yaml')
        with open(args.data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    coco91class = coco80_to_coco91_class()
    dataloader = create_dataloader(data['val'], 1280, 1, 64, args, pad=0.5, rect=False, prefix=colorstr('val'))[0]
    jdict = []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        f = paths[0].split('/')[-1]
        pred = np.fromfile(os.path.join(args.result_path, f.split(".")[0] + '_0.bin'), np.float32)\
            .reshape(1, 102000, 85)
        pred = torch.from_numpy(pred)
        lb = []
        out = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.65, labels=lb, multi_label=True)
        for si, pred in enumerate(out):
            path = Path(paths[si])
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                jdict.append({'image_id': image_id,
                              'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                              'bbox': [round(x, 3) for x in b],
                              'score': round(p[4], 5)})

    anno_json = args.img_path
    pred_json = './yolov7-e6_predictions.json'  # predictions json
    print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        evl = COCOeval(anno, pred, 'bbox')
        if is_coco:
            evl.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        evl.evaluate()
        evl.accumulate()
        evl.summarize()
        ap, ap50, ap75 = evl.stats[:3]
        print(f'ap:{ap}, ap50:{ap50}, ap75:{ap75}')
    except Exception as e:
        print(f'pycocotools unable to run: {e}')
