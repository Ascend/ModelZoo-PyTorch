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


from __future__ import print_function
from __future__ import division


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchreid import models
from torchreid.data_manager import DatasetManager
from torchreid.dataset_loader import ImageDataset
from torchreid.eval_metrics import evaluate
from torchreid import transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')
import os

def postprocess(ranks=range(1, 51)):

    dataset = DatasetManager(dataset_dir=args.dataset, root=args.root)

    transform_test = T.Compose_Keypt([
        T.Resize_Keypt((256,256)),
        T.ToTensor_Keypt(),
        T.Normalize_Keypt(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    queryloader = DataLoader(
        ImageDataset(dataset.query, keyptaware=False, heatmapaware=False,
                     segmentaware=False,
                     transform=transform_test, imagesize=(256,256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, keyptaware=False, heatmapaware=False,
                     segmentaware=False,
                     transform=transform_test, imagesize=(256,256)),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        drop_last=False,
    )
    qf = []
    q_vids = []
    q_camids = []
    for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(queryloader):
        q_vids.extend(vids)
        q_camids.extend(camids)
    q_vids = np.asarray(q_vids)
    q_camids = np.asarray(q_camids)

    for root, folder, files in os.walk(args.queryfeature_path):
        files.sort(key=lambda x:int(x.split('_')[0]))
        for file in files:
            truefile1 = file.split('_')[1]
            if truefile1 == "4.bin": #将benckmark推理出的第四个输出"features"读入，features为计算mAP值的特征
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    featuresq = np.fromfile(f, dtype="float32")
                    featuresq = torch.from_numpy(featuresq)
                    featuresq = featuresq.unsqueeze(0)
                    featuresq = featuresq.data.cpu()
                    qf.append(featuresq)
    qf = torch.cat(qf, 0)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf = []
    g_vids = []
    g_camids = []
    for batch_idx, (imgs, vids, camids, vcolors, vtypes, vkeypts) in enumerate(galleryloader):
        g_vids.extend(vids)
        g_camids.extend(camids)
    g_vids = np.asarray(g_vids)
    g_camids = np.asarray(g_camids)

    for root, folder, files in os.walk(args.galleryfeature_path):
        files.sort(key=lambda x: int(x.split('_')[0]))
        for file in files:
            truefile2 = file.split('_')[1]
            if truefile2== "4.bin": #将benckmark推理出的第四个输出"features"读入，features为计算mAP值的特征
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    featuresg = np.fromfile(f, dtype="float32")
                    featuresg = torch.from_numpy(featuresg)
                    featuresg = featuresg.unsqueeze(0)
                    featuresg = featuresg.data.cpu()
                    gf.append(featuresg)
    gf = torch.cat(gf, 0)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_vids, g_vids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", default="/opt/npu/veri/image_query")
    parser.add_argument("--gallery_dir", default="/opt/npu/veri/image_test")
    parser.add_argument("--queryfeature_path", default="./result/dumpOutput_device0_query")
    parser.add_argument("--galleryfeature_path", default="./result/dumpOutput_device0_gallery")
    parser.add_argument('--root', type=str, default='/opt/npu/',
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, default='veri',
                        help="name of the dataset")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--test-batch', default=1, type=int,
                        help="test batch size")
    parser.add_argument('-a', '--arch', type=str, default='densenet121', choices=models.get_names())
    args = parser.parse_args()

    postprocess()
