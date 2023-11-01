# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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

import yaml
import json
import time
import argparse
import torch
import torch_aie
from torch_aie import _enums

from ais_bench.infer.interface import InferSession

from yolov3.utils.datasets import create_dataloader
from yolov3.utils.general import scale_coords, non_max_suppression
from yolov3.common.util.dataset import evaluate, coco80_to_coco91_class, correct_bbox, save_coco_json


def forward_nms_script(model, dataloader, cfg):
    pred_results = []
    num = 0
    performance = 0
    for (img, targets, paths, shapes) in tqdm(dataloader):
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        padding = False

        img = torch.Tensor(img)
        result, Performance = pt_infer(model, img)
        performance += Performance
        num += 1
        if len(result) == 3:  # number of output nodes is 3, each shape is (bs, na, no, ny, nx)
            out = []
            for i in range(len(result)):
                anchors = torch.tensor(cfg['anchors'])
                stride = torch.tensor(cfg['stride'])
                cls_num = cfg['class_num']
                if padding == True:
                    result[i] = result[i][:nb]
                correct_bbox(result[i], anchors[i], stride[i], cls_num, out)
            box_out = torch.cat(out, 1)
        else:  # only use the first output node, which shape is (bs, -1, no)
            if padding == True:
                result[0] = result[0][:nb]
            box_out = torch.tensor(result[0])

        # non_max_suppression
        boxout = nms(box_out, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        for idx, pred in enumerate(boxout):
            try:
                scale_coords(img[idx].shape[1:], pred[:, :4], shapes[idx][0], shapes[idx][1])  # native-space pred
            except:
                pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # append to COCO-JSON dictionary
            path = Path(paths[idx])
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            save_coco_json(pred, pred_results, image_id, coco80_to_coco91_class())
    print('性能(微秒)：', performance / num)
    return pred_results

def pt_infer(model, input_li):
    T1 = time.perf_counter()
    results = model.forward(input_li)
    T2 = time.perf_counter()
    return results, T2 - T1

def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres)

    return boxout


def main(opt, cfg):
    # load model
    model_orig = torch.jit.load(opt.model)
    min_shape = (1, 3, 640, 640)
    max_shape = (32, 3, 640, 640)
    torch_aie.set_device(0)
    inputs = []
    inputs.append(torch_aie.Input(min_shape = min_shape, max_shape= max_shape))
    model = torch_aie.compile(
        model_orig,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False,
        allow_tensor_replace_int=False,
        min_block_size=3,
        torch_executed_ops=[],
        soc_version="Ascend310P3",
        optimization_level=0)


    # load dataset
    single_cls = False if opt.tag == '9.6.0' else opt
    dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size, max(cfg["stride"]), single_cls, pad=0.5)[0]
    # inference & nms
    pred_results = forward_nms_script(model, dataloader, cfg)

    pred_json_file = f"{opt.model.split('.')[0]}_{opt.tag}_predictions.json"
    print(f'saving results to {pred_json_file}')
    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)

    # evaluate mAP
    evaluate(opt.ground_truth_json, pred_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv3 offline model inference.')
    parser.add_argument('--data_path', type=str, default="coco", help='root dir for val images and annotations')
    parser.add_argument('--ground_truth_json', type=str, default="coco/instances_val2017.json",
                        help='annotation file path')
    parser.add_argument('--tag', type=str, default='9.6.0', help='yolov3 tags')
    parser.add_argument('--model', type=str, default="yolov3_torch_aie.pt", help='ts model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)
