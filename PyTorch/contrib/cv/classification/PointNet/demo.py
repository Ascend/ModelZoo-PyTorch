# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=32)
parser.add_argument('--model', type=str, default='./checkpoint_79_epoch.pkl')
parser.add_argument('--dataset', type=str, default="./data/shapenetcore_partanno_segmentation_benchmark_v0",
                    help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', type=bool, default=True, help="use feature transform")

opt = parser.parse_args()

if opt.dataset_type == 'shapenet':
    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'modelnet40':
    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

num_classes = len(test_dataset.classes)
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(opt.model, map_location='cpu')['model_state_dict'])

classifier.eval()

with torch.no_grad():
    total_correct = 0
    test_set = 0
    error = 0
    data = next(iter(testdataloader))

    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    output, _, _ = classifier(points)
    _, pred = output.topk(1, 1, True, True)
    result = torch.argmax(output, 1)
    print("class: ", pred[0][0].item())
    print(result)
