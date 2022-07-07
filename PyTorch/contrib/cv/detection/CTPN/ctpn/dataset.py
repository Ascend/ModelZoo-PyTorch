# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
import os
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from ctpn.utils import cal_rpn

IMAGE_MEAN = [123.68, 116.779, 103.939]

'''
从xml文件中读取图像中的真值框
'''


def readxml(path):
    gtboxes = []
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))
                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes)


'''
读取VOC格式数据，返回用于训练的图像、anchor目标框、标签
'''


class VOCDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def generate_gtboxes(self, xml_path, rescale_fac=1.0):
        base_gtboxes = readxml(xml_path)
        gtboxes = []
        for base_gtbox in base_gtboxes:
            xmin, ymin, xmax, ymax = base_gtbox
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                _next = 16 * i - 0.5
                gtboxes.append((prev, ymin, _next, ymax))
                prev = _next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        rescale_fac = max(h, w) / 1000
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w, h))

        xml_path = os.path.join(self.labelsdir, img_name.split('.')[0] + '.xml')
        gtbox = self.generate_gtboxes(xml_path, rescale_fac)

        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr] = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        m_img = img - IMAGE_MEAN
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


################################################################################


class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac=1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            gtboxes.append((xmin, ymin, xmax, ymax))
        return np.array(gtboxes)

    def box_transfer_v2(self, coor_lists, rescale_fac=1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                _next = 16 * i - 0.5
                gtboxes.append((prev, ymin, _next, ymax))
                prev = _next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def parse_gtfile(self, gt_path, rescale_fac=1.0):
        coor_lists = list()
        with open(gt_path, 'r', encoding="utf-8-sig") as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists, rescale_fac)

    def draw_boxes(self, img, cls, base_anchors, gt_box):
        for i in range(len(cls)):
            if cls[i] == 1:
                pt1 = (int(base_anchors[i][0]), int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]), int(base_anchors[i][3]))
                img = cv2.rectangle(img, pt1, pt2, (200, 100, 100))
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]), int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]), int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))
        return img

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)[:, :, ::-1]

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1000
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w, h))

        # gt_path = os.path.join(self.labelsdir, img_name.split('.')[0]+'.txt')
        gt_path = os.path.join(self.labelsdir, "gt_" + img_name.split('.')[0] + '.txt')
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # random flip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        print(gtbox)
        [cls, regr] = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        m_img = img - IMAGE_MEAN
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


'''
读取ICDAR格式数据，返回用于训练的图像、anchor目标框、标签
'''


class icdarDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir
        self.short_len = 1008
        self.long_len = 1008

    def __len__(self):
        return len(self.img_names)

    def generate_gtboxes(self, txt_path, w_pad, h_pad, rescale_fac):

        coor_lists = list()
        with open(txt_path, 'r', encoding="utf-8-sig") as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(' ')[:4]
                for i in range(len(coor_list)):
                    coor_list[i] = int(coor_list[i])
                coor_lists.append(coor_list)
        base_gtboxes = np.array(coor_lists)

        gtboxes = []
        for base_gtbox in base_gtboxes:
            xmin, ymin, xmax, ymax = base_gtbox
            xmin = int(xmin / rescale_fac) + w_pad
            xmax = int(xmax / rescale_fac) + w_pad
            ymin = int(ymin / rescale_fac) + h_pad
            ymax = int(ymax / rescale_fac) + h_pad
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                _next = 16 * i - 0.5
                gtboxes.append((prev, ymin, _next, ymax))
                prev = _next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)[:, :, ::-1]
        h, w, c = img.shape
        scale_range = random.uniform(0.7, 1.5)
        h = h * scale_range
        w = w * scale_range
        rescale_fac = 1.0
        if min(h, w) > self.short_len or max(h, w) > self.long_len:
            rescale_fac = max(min(h, w) / self.short_len, max(h, w) / self.long_len)
        h = int(round(h / rescale_fac))
        w = int(round(w / rescale_fac))
        rescale_fac = rescale_fac / scale_range
        img = cv2.resize(img, (w, h))
        if h > w:
            target_w = self.short_len
            target_h = self.long_len
        else:
            target_w = self.long_len
            target_h = self.short_len

        w_pad = random.randint(0, target_w - w)
        h_pad = random.randint(0, target_h - h)

        w_pad2 = target_w - w - w_pad
        h_pad2 = target_h - h - h_pad
        img = np.pad(img, ((h_pad, h_pad2), (w_pad, w_pad2), (0, 0)))

        txt_path = os.path.join(self.labelsdir, 'gt_' + img_name.split('.')[0] + '.txt')
        gtbox = self.generate_gtboxes(txt_path, w_pad, h_pad, rescale_fac)
        h, w = target_h, target_w
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2
        [cls, regr] = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        m_img = img - IMAGE_MEAN
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr
