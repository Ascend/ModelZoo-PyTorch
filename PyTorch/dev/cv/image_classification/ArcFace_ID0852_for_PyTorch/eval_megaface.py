#!/usr/bin/env python
# encoding: utf-8
#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
'''
@author: wujiyang
@contact: wujiyang@hust.edu.cn
@file: eval_megaface.py
@time: 2018/12/24 16:28
@desc: megaface feature extractor
'''
import numpy as np
import struct
import os
import torch.utils.data
from backbone import mobilefacenet, cbam, self_attention
from dataset.megaface import MegaFace
import torchvision.transforms as transforms
import argparse
from torch.nn import DataParallel


cv_type_to_dtype = {5: np.dtype('float32'), 6: np.dtype('float64')}
dtype_to_cv_type = {v: k for k, v in cv_type_to_dtype.items()}

def write_mat(filename, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])

    with open(filename, 'wb') as outfile:
        outfile.write(header)
        outfile.write(m.data)


def read_mat(filename):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    with open(filename, 'rb') as fin:
        rows, cols, stride, type_ = struct.unpack('iiii', fin.read(4 * 4))
        mat = np.fromstring(str(fin.read(rows * stride)), dtype=cv_type_to_dtype[type_])
        return mat.reshape(rows, cols)


def extract_feature(model_path, backbone_net, face_scrub_path, megaface_path, batch_size=32, gpus='0', do_norm=False):

    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(args.backbone, ' is not available!')

    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('npu' if torch.npu.is_available() else 'cpu')

    net.load_state_dict(torch.load(model_path)['net_state_dict'])
    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)
    net.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    megaface_dataset = MegaFace(face_scrub_path, megaface_path, transform=transform)
    megaface_loader = torch.utils.data.DataLoader(megaface_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=12, drop_last=False)

    for data in megaface_loader:
        img, img_path= data[0].to(device), data[1]
        with torch.no_grad():
            output = net(img).data.cpu().numpy()

        if do_norm is False:
            for i in range(len(img_path)):
                abs_path = img_path[i] + '.feat'
                write_mat(abs_path, output[i])
            print('extract 1 batch...without feature normalization')
        else:
            for i in range(len(img_path)):
                abs_path = img_path[i] + '.feat'
                feat = output[i]
                feat = feat / np.sqrt((np.dot(feat, feat)))
                write_mat(abs_path, feat)
            print('extract 1 batch...with feature normalization')
    print('all images have been processed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', type=str, default='./model/RES100_RES100_IR_20190423_100728/Iter_333000_net.ckpt', help='The path of trained model')
    parser.add_argument('--backbone_net', type=str, default='CBAM_100', help='MobileFace, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--facescrub_dir', type=str, default='/media/sda/megaface_test_kit/facescrub_align_112/', help='facescrub data')
    parser.add_argument('--megaface_dir', type=str, default='/media/sda/megaface_test_kit/megaface_align_112/', help='megaface data')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='gpu list')
    parser.add_argument("--do_norm", type=int, default=1, help="1 if normalize feature, 0 do nothing(Default case)")
    args = parser.parse_args()

    extract_feature(args.model_path, args.backbone_net, args.facescrub_dir, args.megaface_dir, args.batch_size, args.gpus, args.do_norm)