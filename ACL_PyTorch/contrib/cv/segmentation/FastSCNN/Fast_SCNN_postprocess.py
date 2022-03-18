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
import sys
sys.path.append('./SegmenTron')
from segmentron.utils.score import SegmentationMetric
import os
import struct
"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np


__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        # self.distributed = distributed
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            self.total_correct += correct.item()
            self.total_label += labeled.item()
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union
        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """

        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output, 1).to(torch.int32) + 1
    target = target + 1
    pixel_labeled = torch.sum(target > 0)
    pixel_correct = torch.sum((predict == target).float() * (target > 0).float())
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1).to(torch.int32) + 1
    target = target.float() + 1
    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.

    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):

    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct

def compute_score(hist, correct, labeled):

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    # freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc


class postprocess(object):
    def __init__(self, result_bin_root):
        self.result_bin_root = result_bin_root
    def process(self):
        sum_mIoU = 0
        sum_pixAcc = 0
        metric = SegmentationMetric(19)
        result_bin_root = self.result_bin_root
        label_bin_root = 'prep_dataset/datasets/gtFine'
        output_path, target_path = _get_output_target_path(result_bin_root, label_bin_root)

        for i in range(len(output_path)):
            output, target = file2tensor(output_path[i], target_path[i])
            metric.update(output, target)
            pixAcc, mIoU = metric.get()
            sum_mIoU += mIoU
            sum_pixAcc += pixAcc
            print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
        print('AvgmIou',sum_mIoU/5, '  |  ', 'AvgpixAcc', sum_pixAcc/5)

def _get_output_target_path(bin_folder, laber_folder):
    result_paths3 = []
    labels_paths = []
    print("result_bin_folder:", bin_folder)
    print('label_bin_folder:', laber_folder)
    for root, _, files in os.walk(bin_folder):
        for file in files:
            if file.endswith('_1.bin'):
                result_path = os.path.join(root, file)
                result_paths3.append(result_path.replace('_1.bin', '_3.bin'))
                temp_path = file.replace('_1.bin', '.bin')
                temp_path = temp_path.replace('leftImg8bit', 'gtFine_labelIds')
                mask_path = os.path.join(laber_folder,temp_path)
                labels_paths.append(mask_path)
    return result_paths3, labels_paths



def file2tensor( bin_file3, mask_path):
    size = os.path.getsize(bin_file3)
    res3 = []
    L = int(size / 2)  # 由于需要的是float32类型，所以按照4字节读取；根据实际情况按字节读取
    binfile = open(bin_file3, 'rb')
    for i in range(L):
        data = binfile.read(2)
        num = struct.unpack('h', data)
        res3.append(num[0])
    binfile.close()

    dim_res = np.array(res3).reshape(1, 19, 1024, 2048)  # 转换为对应的shape，可通过在线推理打印outputs的shape获取到
    tensor_res3 = torch.tensor(dim_res, dtype=torch.float16)
    # print(bin_file, tensor_res.dtype, tensor_res.shape)

    size = os.path.getsize(mask_path)
    mask = []
    L = int(size / 4)  # 由于需要的是float32类型，所以按照4字节读取；根据实际情况按字节读取
    binfile = open(mask_path, 'rb')
    for i in range(L):
        data = binfile.read(4)
        num = struct.unpack('f', data)
        mask.append(num[0])
    binfile.close()

    dim_res = np.array(mask).reshape(1, 1, 1024, 2048)  # 转换为对应的shape，可通过在线推理打印outputs的shape获取到
    target = torch.tensor(dim_res, dtype=torch.float32)
    # print(bin_file, tensor_res.dtype, tensor_res.shape)

    return tensor_res3,target

if __name__ == '__main__':
 result_path = sys.argv[1]
 pro_process = postprocess(result_path)
 pro_process.process()


