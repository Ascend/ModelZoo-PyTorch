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

import os
import argparse
import cv2
import pickle
import numpy as np
import tqdm


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_annotation", default="./origin_pictures.info")
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0/")
    parser.add_argument("--det_results_path", default="./detection-results/")
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
    from detectron2.layers.mask_ops import paste_masks_in_image
    import torch.nn.functional as F
    from detectron2.evaluation import COCOEvaluator
    from detectron2.structures import Boxes, Instances
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import logging
    from detectron2.modeling.postprocessing import detector_postprocess

    logging.basicConfig(level=logging.INFO)
    evaluator = COCOEvaluator('coco_2017_val')
    evaluator.reset()
    coco_class_map = {id:name for id, name in enumerate(MetadataCatalog.get('coco_2017_val').thing_classes)}

    cnt = 0
    inputs = DatasetCatalog.get('coco_2017_val')
    outputs = []
    for bin_file in tqdm.tqdm(sorted(total_img)):
        cnt = cnt + 1
        path_base = os.path.join(bin_path, bin_file)
        res_buff = []
        for num in range(0, flags.net_out_num):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 0:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 4])
                elif num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    buf = np.reshape(buf, [100, 1])
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="int64")
                    buf = np.reshape(buf, [100, 1])
                elif num == 3:
                    bboxes = np.fromfile(path_base + "_" + str(num - 3) + ".bin", dtype="float32")
                    bboxes = np.reshape(bboxes, [100, 4])
                    bboxes = torch.from_numpy(bboxes)
                    scores = np.fromfile(path_base + "_" + str(num - 2) + ".bin", dtype="float32")
                    scores = np.reshape(scores, [100, 1])
                    scores = torch.from_numpy(scores)
                    labels = np.fromfile(path_base + "_" + str(num - 1) + ".bin", dtype="int64")
                    labels = np.reshape(labels, [100, 1])
                    labels = torch.from_numpy(labels)
                    mask_pred = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")
                    mask_pred = np.reshape(mask_pred, [100, 80, 28, 28])
                    mask_pred = torch.from_numpy(mask_pred)

                    org_img_size = img_size_dict[bin_file][:2]
                    result = Instances((org_img_size[1], org_img_size[0]))

                    predbox = postprocess_bboxes(bboxes, org_img_size, flags.net_input_height, flags.net_input_width)

                    result.pred_boxes = Boxes(predbox)
                    result.scores = scores.reshape([100])
                    result.pred_classes = labels.reshape([100])

                    # (N, 80, M, M) -> (N, 1, M, M)
                    mask_pred = mask_pred[range(len(mask_pred)), labels[:, 0]][:, None]
                    # (N, 1, M, M) -> (N, M, M)
                    mask_pred = mask_pred[:, 0, :, :]

                    masks = paste_masks_in_image(mask_pred, result.pred_boxes, (org_img_size[1], org_img_size[0]), 0.5)
                    device = masks.device if isinstance(masks, torch.Tensor) else torch.device("cpu")
                    masks = torch.as_tensor(masks, dtype=torch.bool)
                    result.pred_masks = masks
                    
                    outputs.append({"instances": result})
                    if (cnt % 100) == 0:
                        evaluator.process(inputs[cnt-100:cnt], outputs)
                        outputs.clear()

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
                continue
            if class_idx < 0 or class_idx > 80:
                continue

            class_name = coco_class_map[int(class_idx)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])

            if flags.ifShowDetObj == True:
                imgCur = cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 2)
                imgCur = cv2.putText(imgCur, class_name, (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if flags.ifShowDetObj == True:
            cv2.imwrite(os.path.join(det_results_path, bin_file +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        det_results_file = os.path.join(det_results_path, bin_file + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)
    
    evaluator.evaluate()