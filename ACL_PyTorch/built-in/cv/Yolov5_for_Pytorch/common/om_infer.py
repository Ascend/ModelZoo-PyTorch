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

import os
import cv2
import glob
import json
import argparse
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from util.acl_net import Net

np.set_printoptions(suppress=True)
neth, netw = 640, 640


class BatchDataLoader:
    def __init__(self, data_path_list: list, batch_size: int):
        self.data_path_list = data_path_list
        self.sample_num = len(data_path_list)
        self.batch_size = batch_size

    def __len__(self):
        return self.sample_num // self.batch_size + int(self.sample_num % self.batch_size > 0)

    @staticmethod
    def read_data(img_path):
        basename = os.path.basename(img_path)
        img0 = cv2.imread(img_path)
        imgh, imgw = img0.shape[:2]
        img = letterbox(img0, new_shape=(neth, netw))[0]  # padding resize
        imginfo = np.array([neth, netw, imgh, imgw], dtype=np.float16)
        return img0, img, imginfo, basename

    def __getitem__(self, item):
        if (item + 1) * self.batch_size <= self.sample_num:
            slice_end = (item + 1) * self.batch_size
            pad_num = 0
        else:
            slice_end = self.sample_num
            pad_num = (item + 1) * self.batch_size - self.sample_num

        img0 = []
        img = []
        img_info = []
        name_list = []
        for path in self.data_path_list[item * self.batch_size:slice_end]:
            i0, x, info, name = self.read_data(path)
            img0.append(i0)
            img.append(x)
            img_info.append(info)
            name_list.append(name)
        valid_num = len(img)
        for _ in range(pad_num):
            img.append(img[0])
            img_info.append(img_info[0])
        return valid_num, name_list, img0, np.stack(img, axis=0), np.stack(img_info, axis=0)


def coco80_to_coco91_class():
    # converts 80-index (val2014/val2017) to 91-index (paper)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    # convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=botttom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def read_class_names(ground_truth_json):
    with open(ground_truth_json, 'r') as file:
        content = file.read()
    content = json.loads(content)
    categories = content.get('categories')
    
    names = {}
    # generate names file
    with open(args.names_path, 'w') as f:
        for id, category in enumerate(categories):
            category_name = category.get('name')
            if len(category_name.split()) == 2:
                temp = category_name.split()
                category_name = temp[0] + '_' + temp[1]
            names[id] = name.strip('\n')
    return names


def draw_bbox(bbox, img0, color, wt, names):
    det_result_str = ''
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0] + 64), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        det_result_str += '{} {} {} {} {} {}\n'.format(
            names[bbox[idx][5]], str(bbox[idx][4]), bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3])
    return img0


def eval(ground_truth_json, detection_results_json):
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    print('Start evaluate *%s* results...' % (annType))
    cocoGt_file = ground_truth_json
    cocoDt_file = detection_results_json
    cocoGt = COCO(cocoGt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)
    imgIds = cocoGt.getImgIds()
    print('get %d images' % len(imgIds))
    imgIds = sorted(imgIds)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # copy-paste style
    eval_results = OrderedDict()
    metric = annType
    metric_items = [
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    ]
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

    for metric_item in metric_items:
        key = f'{metric}_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results[f'{metric}_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f}'
    )
    print(dict(eval_results))


def main(args):
    if args.visable:
        coco_names = read_class_names(args.ground_truth_json)
    coco91class = coco80_to_coco91_class()
    model = Net(device_id=args.device_id, model_path=args.model)

    det_result_dict = []
    if not os.path.exists('output'):
        os.mkdir('output')
    img_path_list = glob.glob(args.img_path + '/*.jpg')
    img_path_list.sort()
    dataloader = BatchDataLoader(img_path_list, args.batch_size)
    total_time = 0.0
    it = 0
    for i in tqdm(range(len(dataloader))):
        it += 1
        valid_num, basename_list, img0_list, img, imginfo = dataloader[i]
        img = img[..., ::-1].transpose(0, 3, 1, 2)  # BGR tp RGB
        image_np = np.array(img, dtype=np.float32)
        image_np_expanded = image_np / 255.0
        img = np.ascontiguousarray(image_np_expanded).astype(np.float16)

        result, dt = model([img, imginfo])  # net out, infer time
        batch_boxout, boxnum = result
        total_time += dt

        for idx in range(valid_num):
            basename = basename_list[idx]
            name, postfix = basename.split('.')
            num_det = int(boxnum[idx][0])
            boxout = batch_boxout[idx][:num_det * 6].reshape(6, -1).transpose().astype(np.float32)  # 6xN -> Nx6
            # convert to coco style
            image_id = int(name)
            box = xyxy2xywh(boxout[:, :4])
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(boxout.tolist(), box.tolist()):
                det_result_dict.append({'image_id': image_id,
                                        'category_id': coco91class[int(p[5])],
                                        'bbox': [round(x, 3) for x in b],
                                        'score': round(p[4], 5)})

            if args.visable:
                img_dw = draw_bbox(boxout, img0_list[idx], (0, 255, 0), 2, coco_names)
                cv2.imwrite(os.path.join('output', basename), img_dw)

    print('model infer average time:{:.3f} ms / {} image'.format(total_time * 1000 / it, args.batch_size))

    print('\nsaveing %s...' % args.detection_results_json)
    with open(args.detection_results_json, 'w') as f:
        json.dump(det_result_dict, f)
    
    eval(args.ground_truth_json, args.detection_results_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YoloV5 offline model inference.')
    parser.add_argument('--detection_results_json', type=str, default="./predictions.json",
                        help='detection result file path')
    parser.add_argument('--ground_truth_json', type=str, default="./instances_val2017.json",
                         help='annotation file path')
    parser.add_argument('--img-path', type=str, default="./val2017", help='input images dir')
    parser.add_argument('--model', type=str, default="yolov5s.om", help='om model path')
    parser.add_argument('--batch-size', type=int, default=1, help='om batch size')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('-v', '--visable', action='store_true',
                        help='draw detect result at image and save to dir \'output\'')
    flags = parser.parse_args()

    main(flags)
