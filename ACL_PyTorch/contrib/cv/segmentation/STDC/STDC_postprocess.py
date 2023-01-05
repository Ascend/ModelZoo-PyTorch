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

import numpy as np
import torch
import argparse
import os
from PIL import Image
from tqdm import tqdm


class GTFineFile(object):
    """
    directory: path to gtFine
    suffix: suffix of the gtFine
    :return path List of gtFine files
    """
    def __init__(self, directory, suffix='_gtFine_labelTrainIds.png'):
        gtFine_list = []
        for root, sub_dirs, files in os.walk(directory):
            for special_file in files:
                if special_file.endswith(suffix):
                    gtFine_list.append(os.path.join(root, special_file))
        self.gtFine_list = gtFine_list

    def get_file(self, filename):
        """ return file path list """
        for file in self.gtFine_list:
            if file.endswith(filename):
                return file


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    label = torch.from_numpy(label)

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
    return [area_intersect, area_union, area_pred_label, area_label]


class IntersectAndUnion(object):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

    Returns:
        iou
        acc
    """

    def __init__(self, num_classes, ignore_index, label_map=dict(), reduce_zero_label=False):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_map = label_map
        self.reduce_zero_label = reduce_zero_label
        self.total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
        self.total_area_label = torch.zeros((num_classes,), dtype=torch.float64)

    def update(self, output, gt_seg_map):
        """ update  """
        [area_intersect, area_union, area_pred_label, area_label] = \
            intersect_and_union(
                output, gt_seg_map, self.num_classes, self.ignore_index,
                self.label_map, self.reduce_zero_label)
        self.total_area_intersect += area_intersect.to(torch.float64)
        self.total_area_union += area_union.to(torch.float64)
        self.total_area_pred_label += area_pred_label.to(torch.float64)
        self.total_area_label += area_label.to(torch.float64)

    def get(self):
        """ get result """
        iou = self.total_area_intersect / self.total_area_union
        acc = self.total_area_intersect / self.total_area_label
        all_acc = self.total_area_intersect.sum() / self.total_area_label.sum()
        mIoU = np.round(np.nanmean(iou) * 100, 2)
        aAcc = np.round(np.nanmean(all_acc) * 100, 2)
        return {'aAcc': aAcc, 'mIoU': mIoU}


def eval_metrics(output_path,
                 gt_path,
                 out_suffix='_leftImg8bit_0.bin',
                 gt_suffix='_gtFine_labelTrainIds.png',
                 result_path='./postprocess_result',
                 num_classes=19,
                 ignore_index=255,
                 label_map=None,
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # init metric
    metric = IntersectAndUnion(num_classes, ignore_index, label_map, reduce_zero_label)
    # init gtFine files list
    fileFinder = GTFineFile(gt_path)

    for root, sub_dirs, files in os.walk(output_path):
        files = [file for file in files if file.endswith('bin')]
        len = str(files.__len__())
        for i, output_name in tqdm(enumerate(files)):
            if not output_name.endswith('bin'):
                continue
            #print('STDC metric [' + str(i + 1) + '/' + len + '] on process: ' + output_name)
            seg_map_name = output_name.replace(out_suffix, gt_suffix)
            seg_map_path = fileFinder.get_file(seg_map_name)

            if seg_map_name is not None:
                seg_map = Image.open(seg_map_path)
                seg_map = np.array(seg_map, dtype=np.uint8)

                output_path = os.path.realpath(os.path.join(root, output_name))
                output = np.fromfile(output_path, dtype=np.uint32).reshape(1024, 2048)
                output = output.astype(np.uint8)
                metric.update(output, seg_map)
            else:
                print("[ERROR] " + seg_map_name + " not find, check the file or make sure --out_suffix")

    # get result
    result = metric.get()
    print(result)
    with open(result_path + '.txt', 'w') as f:
        f.write('aAcc: {}\n'.format(result['aAcc']))
        f.write('mIoU: {}\n'.format(result['mIoU']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('mIoU calculate')
    parser.add_argument('--output_path', default="./result",
                        help='path to om/onnx output file, default ./result')
    parser.add_argument('--gt_path', default="/opt/npu/cityscapes/gtFine/val",
                        help='path to gtFine/val, default /opt/npu/cityscapes/gtFine/val')
    parser.add_argument('--out_suffix', default="_leftImg8bit_0.bin",
                        help='suffix of the om/onnx output, default "_leftImg8bit_1.bin"')
    parser.add_argument('--result_path', default="./postprocess_result",
                        help='path to save the script result, default ./postprocess_result.txt')

    args = parser.parse_args()

    output_path = os.path.realpath(args.output_path)
    gt_path = os.path.realpath(args.gt_path)
    out_suffix = args.out_suffix
    result_path = os.path.realpath(args.result_path)
    print(output_path)
    print(gt_path)
    eval_metrics(output_path, gt_path, out_suffix=out_suffix, result_path=result_path)