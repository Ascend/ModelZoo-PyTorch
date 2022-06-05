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
import numpy as np
import torch
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess for OCRNet mocel')
    parser.add_argument('--pred_path', default="preds")
    parser.add_argument('--bin_file_path', default="cityscapes_bin", help='cityscapes processed by preprocessor')
    return parser.parse_args()

def read_files(preds_path, labels_path):
    pred_list = [os.path.join(preds_path, file) for file in os.listdir(preds_path)]
    label_list = [os.path.join(labels_path, label) for label in os.listdir(labels_path)]
    files = []
    for pred_path in pred_list:
        id = int(pred_path.split('/')[-1].split('_')[1][3:])
        files.append({
            "id": id,
            "pred_path": pred_path
        })
    files = sorted(files, key=lambda f: f["id"])
    for label_path in label_list:
        id = int(label_path.split('/')[-1].split('.')[0].split('_')[1][3:])
        files[id]["label_path"] = label_path
    return files

def get_pred_mat(pred_path):
    file = open(pred_path)
    pred = []
    for line in file.readlines():
        pred.append(line.strip().split(' '))  # 去除头尾的空白符之后，按照空格分割
    pred = np.asarray(pred).astype(np.int64)
    pred = torch.from_numpy(pred)
    pred = pred.reshape((-1, 1024, 2048))

    return pred


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def get_mean_iou(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def get_accuracy(intersect_area, pred_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def get_kappa(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa


def calculate_area(pred, label, num_classes, ignore_index):
    if len(pred.shape) == 4:
        pred = torch.squeeze(pred, 1)
    if len(label.shape) == 4:
        label = torch.squeeze(label, 1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
            pred.shape, label.shape))
        # Delete ignore_index
    mask = label != ignore_index
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = get_one_hot(pred, num_classes + 1)
    label = get_one_hot(label, num_classes + 1)
    pred = pred[:, :, :, 1:]
    label = label[:, :, :, 1:]

    # print("after one-hot: ", pred[:,:,:,3])

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[:, :, :, i]
        label_i = label[:, :, :, i]
        pred_area_i = torch.sum(pred_i).unsqueeze(0)
        label_area_i = torch.sum(label_i).unsqueeze(0)
        intersect_area_i = torch.sum(pred_i * label_i).unsqueeze(0)
        # print(pred_area_i)
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)
    pred_area = torch.cat(pred_area, 0)
    # print(pred_area)
    label_area = torch.cat(label_area, 0)
    intersect_area = torch.cat(intersect_area, 0)
    return intersect_area, pred_area, label_area


def main(args):
    preds_path = args.pred_path
    labels_path = os.path.join(args.bin_file_path, 'labels')

    files = read_files(preds_path, labels_path)


    num_classes = 19
    ignore_index = 255
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    for i, item in tqdm(enumerate(files)):
        pred_path = item["pred_path"]
        label_path = item["label_path"]
        # label = np.asarray(Image.open(label_path)).astype(np.int64)
        label = np.fromfile(label_path, dtype=np.int32).astype(np.int64)
        label = label.reshape((-1, 1024, 2048))
        # label = label[np.newaxis, :, :]
        label = torch.from_numpy(label)
        pred = get_pred_mat(pred_path)
        intersect_area, pred_area, label_area = calculate_area(
            pred,
            label,
            num_classes,
            ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
        # 记录时间
    class_iou, miou = get_mean_iou(intersect_area_all, pred_area_all,
                                   label_area_all)
    class_acc, acc = get_accuracy(intersect_area_all, pred_area_all)
    kappa = get_kappa(intersect_area_all, pred_area_all, label_area_all)
    print("[EVAL] #mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(
        miou, acc, kappa))
    print("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
    print("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
