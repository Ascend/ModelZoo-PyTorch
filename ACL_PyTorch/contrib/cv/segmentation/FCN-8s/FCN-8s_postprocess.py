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

import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
from terminaltables import AsciiTable

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor')

PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def load_annotations(img_dir, ann_dir, split):
    img_suffix = '.jpg'
    seg_map_suffix = '.png'
    img_infos = []
    if split is not None:
        with open(split) as f:
            for line in f:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
    else:
        for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
            img_infos.append(img_info)

    return img_infos


def get_gt_seg_maps(img_infos, ann_dir):
    """Get ground truth segmentation maps for evaluation."""
    gt_seg_maps = []
    for img_info in img_infos:
        seg_map = osp.join(ann_dir, img_info['ann']['seg_map'])
        gt_seg_map = mmcv.imread(
            seg_map, flag='unchanged', backend='pillow')
        gt_seg_maps.append(gt_seg_map)
    return gt_seg_maps


def voc2012_evaluation(results, gt_seg_maps):
    metric = ['mIoU']
    eval_results = {}

    num_classes = len(CLASSES)
    ignore_index = 255
    label_map = dict()
    reduce_zero_label = False

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
    for i in range(num_imgs):
        if isinstance(results[i], str):
            pred_label = torch.from_numpy(np.load(results[i]))
        else:
            pred_label = torch.from_numpy((results[i]))

        if isinstance(gt_seg_maps[i], str):
            label = torch.from_numpy(
                mmcv.imread(gt_seg_maps[i], flag='unchanged', backend='pillow'))
        else:
            label = torch.from_numpy(gt_seg_maps[i])

        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id
        if reduce_zero_label:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect

        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    iou = total_area_intersect / total_area_union
    ret_metrics.append(iou)
    ret_metrics = [metric.numpy() for metric in ret_metrics]

    class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
    class_names = CLASSES

    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
    ]
    for i in range(num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                          ['m' + head
                           for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in ret_metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                              [ret_metrics_mean[1]] +
                              [ret_metrics_mean[0]])

    print('per class results:')
    table = AsciiTable(class_table_data)
    print('\n' + table.table)
    print('Summary:')
    table = AsciiTable(summary_table_data)
    print('\n' + table.table)

    for i in range(1, len(summary_table_data[0])):
        eval_results[summary_table_data[0]
        [i]] = summary_table_data[1][i] / 100.0
    for idx, sub_metric in enumerate(class_table_data[0][1:], 1):
        for item in class_table_data[1:]:
            eval_results[str(sub_metric) + '.' +
                         str(item[0])] = item[idx] / 100.0
    return eval_results


def postprocess_mask(mask, image_size, net_input_width, net_input_height):
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pad_w = net_input_width - w * scale
    pad_h = net_input_height - h * scale
    pad_left = (pad_w // 2)
    pad_top = (pad_h // 2)
    if pad_top < 0:
        pad_top = 0
    if pad_left < 0:
        pad_left = 0
    pad_left = int(pad_left)
    pad_top = int(pad_top)
    a = int(500 - pad_top)
    b = int(500 - pad_left)
    mask = mask[pad_top:a, pad_left:b]
    import torch.nn.functional as F
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))
    mask = F.interpolate(mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)

    mask = mask.squeeze().to(dtype=torch.int32).numpy()
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./voc12_jpg.info")
    parser.add_argument("--img_dir", default="/opt/npu/VOCdevkit/VOC2012/JPEGImages")
    parser.add_argument("--ann_dir", default="/opt/npu/VOCdevkit/VOC2012/SegmentationClass")
    parser.add_argument("--split", default="/opt/npu/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt")
    parser.add_argument("--net_input_width", default=500)
    parser.add_argument("--net_input_height", default=500)
    args = parser.parse_args()

    # generate dict according to annotation file for query resolution
    # load width and height of input images
    img_size_dict = dict()

    with open(args.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    # read bin file for generate predict result
    bin_path = args.bin_data_path
    total_img = set([name[:name.rfind('_')]for name in os.listdir(bin_path) if "bin" in name])

    res_buff = []
    for bin_file in sorted(total_img):
        path_base = os.path.join(bin_path, bin_file)
        # load all segected output tensor

        output = np.fromfile(path_base + "_" + str(0) + ".bin", dtype="int64")
        output = np.reshape(output, [500, 500])
        current_img_size = img_size_dict[bin_file]
        output = postprocess_mask(output, img_size_dict[bin_file], 500, 500)
        res_buff.append(output)

    seg_result = res_buff
    # ground truth
    img_infos = load_annotations(args.img_dir, args.ann_dir, split=args.split)
    gt_seg_maps = get_gt_seg_maps(img_infos, args.ann_dir)
    seg_result = voc2012_evaluation(seg_result, gt_seg_maps)
    
    
    with open('./voc_seg_result.txt', 'w') as f:
        for key, value in seg_result.items():
            f.write(key + ': ' + str(value) + '\n')
    
