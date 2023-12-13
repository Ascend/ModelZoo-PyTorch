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
import argparse
import cv2
import numpy as np
from tqdm import tqdm
def postprocess_bboxes(bboxes, image_size, net_input_width, net_input_height):
    org_w = image_size[0]
    org_h = image_size[1]

    scale = 800 / min(org_w, org_h)
    new_w = int(np.floor(org_w * scale))
    new_h = int(np.floor(org_h * scale))
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w) * scale

    bboxes[:, 0] = (bboxes[:, 0]) / scale
    bboxes[:, 1] = (bboxes[:, 1]) / scale
    bboxes[:, 2] = (bboxes[:, 2]) / scale
    bboxes[:, 3] = (bboxes[:, 3]) / scale

    return bboxes

def postprocess_masks(masks, image_size, net_input_width, net_input_height):
    org_w = image_size[0]
    org_h = image_size[1]

    scale = 800 / min(org_w, org_h)
    new_w = int(np.floor(org_w * scale))
    new_h = int(np.floor(org_h * scale))
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w) * scale

    pad_w = net_input_width - org_w * scale
    pad_h = net_input_height - org_h * scale
    top = 0
    left = 0
    hs = int(net_input_height - pad_h)
    ws = int(net_input_width - pad_w)

    masks = masks.to(dtype=torch.float32)
    res_append = torch.zeros(0, org_h, org_w)
    if torch.cuda.is_available():
        res_append = res_append.to(device='cuda')
    for i in range(masks.size(0)):
        mask = masks[i][0][top:hs, left:ws]
        mask = mask.expand((1, 1, mask.size(0), mask.size(1)))
        mask = F.interpolate(mask, size=(int(org_h), int(org_w)), mode='bilinear', align_corners=False)
        mask = mask[0][0]
        mask = mask.unsqueeze(0)
        res_append = torch.cat((res_append, mask))

    return res_append[:, None]

import pickle
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_annotation", default="./maskrcnn_jpeg.info")
    parser.add_argument("--bin_data_path", default="./result")
    parser.add_argument("--det_results_path", default="./precision_result")
    parser.add_argument("--net_out_num", type=int, default=4)
    parser.add_argument("--net_input_width", type=int, default=1344)
    parser.add_argument("--net_input_height", type=int, default=1344)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    flags = parser.parse_args()

    img_size_dict = dict()
    with open(flags.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    bin_path = flags.bin_data_path
    det_results_path = flags.det_results_path
    os.makedirs(det_results_path, exist_ok=True)
    total_img = set([name[:name.rfind('_')] for name in os.listdir(bin_path) if "bin" in name])

    import torch
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    import torch.nn.functional as F
    from detectron2.evaluation import COCOEvaluator
    from detectron2.structures import Boxes, Instances
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import logging
    logging.basicConfig(level=logging.INFO)
    evaluator = COCOEvaluator('coco_2017_val')
    evaluator.reset()
    coco_class_map = {id:name for id, name in enumerate(MetadataCatalog.get('coco_2017_val').thing_classes)}
    results = []

    cnt = 0
    for bin_file in tqdm(sorted(total_img)):
        cnt = cnt + 1
        path_base = os.path.join(bin_path, bin_file)
        res_buff = []
        for num in range(0, flags.net_out_num):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 0:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 4])
                elif num == 1:
                    buf = np.fromfile(path_base + "_" + str(num + 2) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 1])
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num - 1) + ".bin", dtype="float32").astype(np.int64)
                    buf = np.reshape(buf, [100, 1])
                elif num == 3:
                    bboxes = np.fromfile(path_base + "_" + str(num - 3) + ".bin", dtype="float32")
                    bboxes = np.reshape(bboxes, [100, 4])
                    bboxes = torch.from_numpy(bboxes)
                    scores = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    scores = np.reshape(scores, [100, 1])
                    scores = torch.from_numpy(scores)
                    labels = np.fromfile(path_base + "_" + str(num - 2) + ".bin", dtype="float32").astype(np.int64)
                    labels = np.reshape(labels, [100, 1])
                    labels = torch.from_numpy(labels)

                    org_img_size = img_size_dict[bin_file][:2]
                    result = Instances((org_img_size[1], org_img_size[0]))


                    img_shape = (flags.net_input_height, flags.net_input_width)

                    
                    '''masks = masks.numpy()
                    img = masks[0]
                    from PIL import Image
                    for j in range(len(masks)):
                        mask = masks[j]
                        mask = mask.astype(bool)
                        img[mask] = img[mask] + 1
                    imag = Image.fromarray((img * 255).astype(np.uint8))
                    imag.save(os.path.join('.', bin_file + '.png'))'''

                    predbox = postprocess_bboxes(bboxes, org_img_size, flags.net_input_height, flags.net_input_width)
                    result.pred_boxes = Boxes(predbox)
                    result.scores = scores.reshape([100])
                    result.pred_classes = labels.reshape([100])

                    results.append({"instances": result})

                res_buff.append(buf)
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")

        current_img_size = img_size_dict[bin_file]
        res_bboxes = np.concatenate(res_buff, axis=1)
        predbox = postprocess_bboxes(res_bboxes, current_img_size, flags.net_input_width, flags.net_input_height)

        if flags.ifShowDetObj == True:
            imgCur = cv2.imread(current_img_size[2])

        det_results_str = ''
        for idx, class_idx in enumerate(predbox[:, 5]):
            if float(predbox[idx][4]) < float(0.05):
            #if float(predbox[idx][4]) < float(0):
                continue
            if class_idx < 0 or class_idx > 80:
                continue

            class_name = coco_class_map[int(class_idx)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])

            if flags.ifShowDetObj == True:
                imgCur = cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 2)
                imgCur = cv2.putText(imgCur, class_name, (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #imgCur = cv2.putText(imgCur, str(predbox[idx][4]), (int(predbox[idx][0]), int(predbox[idx][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if flags.ifShowDetObj == True:
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)

    save_variable(results, './results.txt')

    inputs = DatasetCatalog.get('coco_2017_val')[:5000]
    evaluator.process(inputs, results)
    evaluator.evaluate()

