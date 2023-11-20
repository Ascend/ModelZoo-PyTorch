# Copyright 2022 Huawei Technologies Co., Ltd
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

import torch
import torch_aie
import numpy as np
import time
import os
import cv2
from tqdm import tqdm
from torch_aie import _enums
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from pathlib import Path
# from common.util.dataset import coco80_to_coco91_class, correct_bbox, save_coco_json
# from utils.general import non_max_suppression, scale_coords
# from common.util.model import nms


def forward_nms_script(model, dataloader, batchsize, device_id):
    pred_results = []
    inference_time = []
    loop_num = 0
    for img in tqdm(dataloader):
    # for i in range(1):
        # print([i / 255.0 for i in list(bytes(img[0]))])
        # print(torch.tensor(img).shape)
        val_transform = Compose([
            transforms.Resize(96, 96),
            transforms.Normalize(),
            ])

        # img_id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
        # if len(img_id) == 0: continue
        # img = cv2.imread(os.path.join("./pytorch-nested-unet/inputs/dsb2018_96/images_test", img_id + '.png'))
        # augmented = val_transform(image=img)
        # img = augmented['image']
        # img = img.astype('float32') / 255
        # img = img.transpose(2, 0, 1)
        img = torch.tensor(img)
        # img = img.unsqueeze(dim=0)
        # print(img.shape)
        # print(img)
        # val = img[0].decode('utf-8')
        # print(type(val), val)
        # img = img.float()
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # nb, _, height, width = img.shape  # batch size, channels, height, width
        padding = False
        # img = torch.Tensor([i / 255.0 for i in list(bytes(img[0]))])
        # img = torch.Tensor(list(bytes(img[0])))

        # pt infer
        result, inference_time = pt_infer(model, img, device_id, loop_num, inference_time)
        pred_results.append(result)
        loop_num += 1
        # if len(result) == 3:  # number of output nodes is 3, each shape is (bs, na, no, ny, nx)
        #     out = []
        #     for i in range(len(result)):
        #         # anchors = torch.tensor(cfg['anchors'])
        #         # stride = torch.tensor(cfg['stride'])
        #         # cls_num = cfg['class_num']
        #         if padding == True:
        #             result[i] = result[i][:nb]
        #         correct_bbox(result[i], anchors[i], stride[i], cls_num, out)
        #     box_out = torch.cat(out, 1)
        # else:  # only use the first output node, which shape is (bs, -1, no)
        #     if padding == True:
        #         result[0] = result[0][:nb]
        #     box_out = torch.tensor(result[0])

        # non_max_suppression
        # boxout = nms(box_out, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        # for idx, pred in enumerate(boxout):
        #     try:
        #         scale_coords(img[idx].shape[1:], pred[:, :4], shapes[idx][0], shapes[idx][1])  # native-space pred
        #     except:
        #         pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        #     # append to COCO-JSON dictionary
        #     path = Path(paths[idx])
        #     image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        #     save_coco_json(pred, pred_results, image_id, coco80_to_coco91_class())
    # print(batchsize, inference_time)
    # avg_inf_time = sum(inference_time) / len(inference_time) / batchsize * 1000
    # print('性能(毫秒)：', avg_inf_time)
    # print("throughput(fps): ", 1000 / avg_inf_time)

    return pred_results

def pt_infer(model, input_li, device_id, loop_num, inference_time):
    input_npu_li = input_li.to("npu:" + str(device_id))
    stream = torch_aie.npu.Stream("npu:" + str(device_id))
    with torch_aie.npu.stream(stream):
        inf_start = time.time()
        output_npu = model.forward(input_npu_li)
        stream.synchronize()
        inf_end = time.time()
        inf = inf_end - inf_start
        # print(inf)
        if loop_num >= 0:   # use 5 step to warmup
            inference_time.append(inf)
    # if loop_num == 3:
    #     print("111", len(output_npu))
    # results = tuple([output_npu[0].to("cpu"), [i.to("cpu") for i in output_npu[1]]])
    results = output_npu.to("cpu")
    # results = model.forward(input_li)
    print(results)
    # results = chage_results(results)
    # print(results)
    return results, inference_time



# def pt_infer(model, input_li, device_id, loop_num):
#     T1 = time.time()
#     results = model.forward(input_li)
#     T2 = time.time()
#     return results, T2 - T1
