#!/usr/bin/env python3
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os.path as osp
import os
import argparse
import torchreid
from torchreid.data.datasets.image.market1501 import Market1501
from torchreid.utils import load_pretrained_weights
from torchreid import metrics

if not os.path.exists("inference"):
    os.makedirs("inference")
os.system('rm -f inference/*')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# data
parser.add_argument('-d', '--data_path', type=str, default='./reid-data/')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--checkpoint', type=str, default='log/osnet_x1_0_market1501_softmax/model/model.pth.tar-350')
parser.add_argument('--config_file', type=str, default='configs/osnet_x1_0_trained_from_scratch.yaml')

args = parser.parse_args()

os.environ['device'] = args.device

def build_model():
    # Create model
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=751,
        loss="softmax",
        pretrained=False,
        use_gpu=False
    )
    load_pretrained_weights(model, args.checkpoint)
    if os.environ['device'] == 'npu':
        model = model.to("npu:0")
    elif os.environ['device'] == 'gpu':
        model = model.to("cuda:0")
    model.eval()
    return model


def get_raw_data():
    name, root = "market1501", args.data_path

    param = {
        "root": root,
        'sources': ['market1501'],
        'targets': ['market1501'],
        'height': 256,
        'width': 128,
        'transforms': ['random_flip', 'random_crop', 'random_patch'],
        'k_tfm': 1,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
        'use_gpu': False,
        'split_id': 0,
        'combineall': False,
        'load_train_targets': False,
        'batch_size_train': 32,
        'batch_size_test': 32,
        'workers': 4,
        'num_instances': 4,
        'num_cams': 1,
        'num_datasets': 1,
        'train_sampler': "RandomSampler",
        'train_sampler_t': "RandomSampler",
        # image dataset specific
        'cuhk03_labeled': False,
        'cuhk03_classic_split': False,
        'market1501_500k': False,
    }

    datamanager = torchreid.data.ImageDataManager(**param)

    query_loader = datamanager.test_loader['market1501']['query']
    gallery_loader = datamanager.test_loader['market1501']['gallery']

    data = next(iter(query_loader))

    fnames = data['impath']
    pids = data['pid']
    imgs = data['img']
    camids = data['camid']

    img = imgs[24]
    pid = pids[24]
    camid = camids[24]
    fname = fnames[24]
    img = torch.unsqueeze(img, dim=0)

    return datamanager, gallery_loader, img, fname, pid

def parse_data_for_eval(data):
    fnames = data['impath']
    imgs = data['img']
    pids = data['pid']
    camids = data['camid']
    return fnames, imgs, pids, camids

def feature_extraction(model, imgs):
    if os.environ['device'] == 'gpu':
        imgs = imgs.cuda()
    elif os.environ['device'] == 'npu':
        imgs = imgs.npu()
    features = model(imgs)
    features = features.cpu().clone()
    return features

def find_imgs_with_id(id, data_loader):
    f_, pids_, camids_, f_names_ = [], [], [], []
    for batch_idx, data in enumerate(data_loader):
        fnames, imgs, pids, camids = parse_data_for_eval(data)
        for fname, img, pid, camid in zip(fnames, imgs, pids, camids):
            if pid == id:
                img = torch.unsqueeze(img, dim=0)
                f_.append(img)
                pids_.append(pid)
                camids_.append(camid)
                f_names_.append(fname)
    f_ = torch.cat(f_, dim=0)
    return f_, pids_, camids_, f_names_

def feature_extraction_single(model, imgs):
    if os.environ['device'] == 'gpu':
        imgs = imgs.cuda()
    elif os.environ['device'] == 'npu':
        imgs = imgs.npu()
    features = model(imgs)
    features = features.cpu().clone()
    return features

def save_image(fname):
    img_name = osp.basename(fname)
    command = "cp %s ./inference/%s" % (fname, img_name)
    os.system(command)


if __name__ == '__main__':
    data_path = args.data_path
    print("load dataset")
    datamanager, gallery_loader, img, fname, pid = get_raw_data()
    print("find a img in gallery with id %d" % pid)
    imgs_gallery, pids, camids, f_names = find_imgs_with_id(pid.item(), gallery_loader)

    print("build model")
    model = build_model()
    print("extract img feature...")
    img_feature = feature_extraction_single(model, img)
    print("extract gallery feature...")
    # gallery_feature = feature_extraction_single(model, img_gallery)
    gallery_feature = feature_extraction(model, imgs_gallery)

    dist_metric = "euclidean"
    print(
        'Computing distance matrix with metric={} ...'.format(dist_metric)
    )
    distmat = metrics.compute_distance_matrix(img_feature, gallery_feature, dist_metric)
    distmat = distmat.cpu().detach().numpy()
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    index = indices[0][0]
    fname_gallery = f_names[index]

    save_image(fname)
    save_image(fname_gallery)
    print("query img saved to ./inference/%s" % osp.basename(fname))
    print("gallery img saved to ./inference/%s" % osp.basename(fname_gallery))
        