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
from reid import datasets
from reid import models
import os.path as osp
from reid.utils.data import transforms as T
from reid.feature_extraction import extract_cnn_feature
from reid.evaluators import extract_features
from reid.utils.serialization import load_checkpoint, save_checkpoint
import os
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
from reid.utils import to_numpy
import argparse


if not os.path.exists("inference"):
    os.makedirs("inference")
os.system('rm -f inference/*')

parser = argparse.ArgumentParser(description="Softmax loss classification")
# data
parser.add_argument('-d', '--data_path', type=str, default='../data/Market-1501')
parser.add_argument('--device', type=str, default='npu')
parser.add_argument('--checkpoint', type=str, default='logs/market-1501/PCB/checkpoint.pth.tar')
args = parser.parse_args()

os.environ['device'] = args.device

def extract_features_single_img(model, img):
    outputs = extract_cnn_feature(model, img)
    return outputs

def build_model():
    # Create model
    model = models.create("resnet50", num_features=256,
                          dropout=0.5, num_classes=751,cut_at_pooling=False, FCN=True)
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    if os.environ['device'] == 'npu':
        model = model.to("npu:0")
    elif os.environ['device'] == 'gpu':
        model = model.to("cuda:0")
    model.eval()
    return model


def get_raw_data():
    name, root = "market", args.data_path
    dataset = datasets.create(name, root)

    fname, pid, camid = dataset.query[44]

    from PIL import Image
    fpath = osp.join(osp.join(args.data_path, "query"), fname)
    img = Image.open(fpath).convert('RGB')
    return img, dataset, fpath


def pairwise_img_gallery_distance(img_feature, gallery_features, gallery):

    x = img_feature.unsqueeze(0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def get_feature(model, img, dataset):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.RectScale(384, 128),
        T.ToTensor(),
        normalizer,
    ])
    img = test_transformer(img)
    img = np.expand_dims(img, axis=0)

    print("extract single img feature...")
    
    img_feature = extract_features_single_img(model, img)

    print("extract gallery features...")
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=osp.join(dataset.images_dir,dataset.gallery_path),
                     transform=test_transformer),
        batch_size=64, num_workers=8,
        shuffle=False)
    gallery_features, _ = extract_features(model, gallery_loader)
    return img_feature, gallery_features
    



if __name__ == '__main__':
    data_path = args.data_path
    img, dataset, img_fpath = get_raw_data()
    model = build_model()

    img_feature, gallery_features = get_feature(model, img, dataset)
    print("pairwise distance")
    distmat = pairwise_img_gallery_distance(img_feature, gallery_features, dataset.gallery)
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    o_name, o_id, o_cid = dataset.gallery[indices[0][0]]

    img_name = osp.basename(img_fpath)
    img_fpath = osp.join(osp.join(data_path, "query"), img_name)
    o_path = osp.join(osp.join(data_path, "bounding_box_test"), o_name)

    command = "cp %s ./inference/%s" % (img_fpath, img_name)
    os.system(command)
    print("input img (query) saved to ./inference/%s" % img_name)
    command = "cp %s ./inference/%s" % (o_path, o_name)
    os.system(command)
    print("predict img (gallery) saved to ./inference/%s" % o_name)
    
