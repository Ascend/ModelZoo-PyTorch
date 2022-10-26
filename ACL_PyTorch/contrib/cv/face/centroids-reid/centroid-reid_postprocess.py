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

import argparse
from pathlib import Path
import glob
import os.path as osp
import re
import numpy as np
import torch
from datasets import init_dataset 
from train_ctl_model import CTLModel

def get_imagedata_info(data):
    pids, cams = [], []
    for _, pid, camid, *_ in data:
        pids += [pid]
        cams += [camid]
    pids = set(pids)
    cams = set(cams)
    num_pids = len(pids)
    num_cams = len(cams)
    num_imgs = len(data)
    return num_pids, num_imgs, num_cams

def process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.txt'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    dataset = []
    for idx, img_path in enumerate(img_paths):
        pid, camid = map(int, pattern.search(img_path).groups())
        assert 1 <= camid <= 8
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]
        dataset.append((img_path, pid, camid, idx))
    return dataset

def prepare_embedding(path):   
    data = np.loadtxt(path, dtype=np.float32)
    matrix = data.reshape([1,2048])
    output = torch.from_numpy(matrix)
    return output

def prepare_tensor(item):   
    b=np.array(item)
    output = np.expand_dims(b, 0)
    output=torch.from_numpy(output) 
    return output

def run_postprocess(self,args):
    query_dir = osp.join(args.dataset_dir, args.query_path)
    gallery_dir = osp.join(args.dataset_dir, args.gallery_path)
    query = process_dir(query_dir, relabel=False)
    gallery = process_dir(gallery_dir, relabel=False)
    outputs = []  
    for x in enumerate(query+gallery):
        emb, class_labels, camid, idx=x[1]
        emb=prepare_embedding(emb)
        class_labels=prepare_tensor(class_labels)
        camid=prepare_tensor(camid)
        idx=prepare_tensor(idx)
        outputs.append(
            {"emb": emb, "labels": class_labels, "camid": camid, "idx": idx}
        )
    embeddings = torch.cat([x["emb"] for x in outputs]).detach().cpu()
    labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    camids = torch.cat([x["camid"] for x in outputs]).cpu().detach().numpy()
    idx = torch.cat([x["idx"] for x in outputs]).cpu().detach().numpy()
    print(embeddings.shape,labels.shape,camids.shape)
    embeddings, labels, camids = self.validation_create_centroids(
         embeddings, labels, camids,respect_camids=self.hparams.MODEL.KEEP_CAMID_CENTROIDS,
    )   
    self.get_val_metrics(embeddings, labels, camids)
    del embeddings, labels, camids

def main(args):
    model = CTLModel   
    model = model.load_from_checkpoint(
        './models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt',
    )      
    run_postprocess(model,args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLT ACL")
    parser.add_argument("--dataset_dir", default="/home/wxq/DukeMTMC-reID/result/", help="embedding root path", type=str)
    parser.add_argument("--query_path", default="", help="query/2022528_15_44_24_128520", type=str)
    parser.add_argument("--gallery_path", default="", help="gallery/2022528_15_43_9_215073", type=str)
    args = parser.parse_args()    
    main(args)