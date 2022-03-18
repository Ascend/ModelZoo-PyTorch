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

import os
import sys
sys.path.append('./exps/mspn.2xstg.coco/')
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset.attribute import load_dataset
from config import cfg

from dataset.attribute import load_dataset
from dataset.COCO.coco import COCODataset


def preprocess(save_path: str):
    cpu_device = torch.device("cpu")
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])   
    attr = load_dataset(cfg.DATASET.NAME)
    stage='val'
    if cfg.DATASET.NAME == 'COCO':
        Dataset = COCODataset 
    dataset = Dataset(attr, stage, transform)   
    # -------- make data_loader -------- #
    class BatchCollator(object):
        def __init__(self, size_divisible):
            self.size_divisible = size_divisible
        def __call__(self, batch):
            transposed_batch = list(zip(*batch))
            images = torch.stack(transposed_batch[0], dim=0)
            scores = list(transposed_batch[1])
            centers = list(transposed_batch[2])
            scales = list(transposed_batch[3])
            image_ids = list(transposed_batch[4])

            return images, scores, centers, scales, image_ids 

    data_loader = torch.utils.data.DataLoader(
            dataset,batch_size=1,collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), )
    data_loader.ori_dataset = dataset

    data=data_loader
    i = 0
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch
        print("=========",img_ids)
        id=[str(x)for x in img_ids]
        idx="".join(id)
        imgs = imgs.to(cpu_device).numpy()
        imgs.tofile(os.path.join(save_path,'img_' + idx + '_' + str(i)+ ".bin"))
        i += 1   


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path",default="$MSPN_HOME/dataset/COCO")
    args = parser.parse_args()
    COCODataset.cur_dir=os.path.join(args.datasets_path)
    save_path = "./pre_dataset"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    preprocess(save_path)















