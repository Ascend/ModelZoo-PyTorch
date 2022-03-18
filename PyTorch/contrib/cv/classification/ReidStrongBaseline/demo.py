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
import argparse
import os
import sys
import numpy as np
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from modeling import build_model
from data.datasets import  ImageDataset
from data import make_data_loader
from data.transforms import build_transforms
from data.collate_batch import val_collate_fn
from utils.re_ranking import re_ranking
from torch.utils.data import DataLoader
from ignite.metrics import Metric
from ignite.engine import Engine

def create_supervised_evaluator(model, metrics,
                                device=None):
    if device:
        if torch.npu.device_count() > 1 or torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.npu.device_count() >= 1 or torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        print(name, metric)
        metric.attach(engine, name)

    return engine


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)       
        indices = np.argsort(distmat, axis=1)
        match = (g_pids[indices] == q_pids[:, np.newaxis])    
        return match

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Demo")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)  
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    device = cfg.MODEL.DEVICE
    if "npu" in cfg.MODEL.DEVICE:
        model = model.to("npu:0")        
    elif "gpu" in cfg.MODEL.DEVICE:
        model = model.to("cuda:0")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    evaluator.run(val_loader)
    match = evaluator.state.metrics['r1_mAP']
    print('query[0] predict the same ID in gallery correctlyȷ', match[0, :])
    
