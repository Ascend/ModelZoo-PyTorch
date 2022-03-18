# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
import os
import sys
import pickle
import argparse

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as trans
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

from verifacation import evaluate
from model import Backbone, l2_norm


class FinetuneDataset(Dataset):
    def __init__(self, dataset_folder, transform):
        self.ids = next(os.walk(dataset_folder))[1]
        self.label_path = os.path.join(dataset_folder, 'label.txt')
        self.transform = transform

        self.issame_list = self.prepare_labels(self.label_path)
        self.img_paths = self.prepare_imgs(dataset_folder)

    def prepare_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            data_label_list = f.readlines()
        data_label_list = [i.strip().split(' ') for i in data_label_list if len(i) > 4]
        return data_label_list

    def prepare_imgs(self, dataset_folder):
        img_paths = []
        for idx in self.ids:
            img_ids = next(os.walk(os.path.join(dataset_folder, idx)))[2]
            for img_id in img_ids:
                img_paths.append(os.path.join(dataset_folder, idx, img_id))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        sample = self.img_paths[item]
        img = Image.open(sample)
        img = img.convert('RGB')
        img = self.transform(img)
        return item, img


class LFWDataset(Dataset):
    def __init__(self, lfw_bin_path, transform):
        self.bins, self.issame_list = pickle.load(open(lfw_bin_path, 'rb'), encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, item):
        sample = self.bins[item]
        img_np_arr = np.frombuffer(sample, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        img = Image.fromarray(img.astype(np.uint8))
        img = self.transform(img)
        return item, img


def build_model(args, device):
    # build model
    model = Backbone(num_layers=args.net_depth, drop_ratio=0.6, mode=args.net_mode)

    # load weights
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # distributed
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device_id], broadcast_buffers=False)

    return model


def build_data_loader(args):
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.finetune:
        dataset = FinetuneDataset(args.data_path, transform)
    else:
        dataset = LFWDataset(args.data_path, transform)

    embeddings = torch.zeros([dataset.__len__(), 512], dtype=torch.float32)

    if args.distributed:
        ds_sample = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=ds_sample)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    return loader, embeddings, dataset.issame_list


@torch.no_grad()
def evaluation(args, model, loader, embeddings, issame_list, device):
    """
    args
    model
    loader: lfw data loader
    embeddings: empty result
    issame_list: label
    device
    """
    embeddings = embeddings.to(device)

    # ----- evaluation -----
    for idx, img in loader:
        img_flip = torch.flip(img, dims=[3])
        img, img_flip = img.to(device), img_flip.to(device)

        emb_batch = model(img) + model(img_flip)
        outputs = l2_norm(emb_batch).detach()
        embeddings[idx] = outputs
        
    if args.distributed:
        embed_gather_list = [torch.zeros_like(embeddings) for _ in range(args.world_size)]
        dist.all_gather(embed_gather_list, embeddings)
        embeddings = embed_gather_list[0]
        for i in range(1, args.world_size):
            embeddings += embed_gather_list[i]

    embeddings = embeddings.cpu().numpy()

    # ----- compute accuracy -----
    if args.is_master_node:
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame_list, nrof_folds=10)
        print('*'*50)
        print('lfw_accuracy: {}'.format(accuracy.mean()))
        print('best_thresholds: {}'.format(best_thresholds.mean()))
        print('*'*50)


def prepare_parser():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se', type=str)
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=100, type=int)
    parser.add_argument("--weights", help="weights path name", default='./work_space/save/model_ir_se100.pth', type=str)
    parser.add_argument("--data_path", help="lfw bin data path", default='./data/faces_emore/lfw.bin', type=str)
    parser.add_argument("--batch_size", help="eval batch size", default=512, type=int)
    parser.add_argument("--num_workers", help="num of workers", default=8, type=int)

    parser.add_argument("--finetune", help="if finetune dataset", default=0, type=int)

    parser.add_argument("--device_type", help="device_type choice in [npu gpu]", default='npu', type=str)
    parser.add_argument("--device_id", help="device_id", default=0, type=int)

    parser.add_argument("--distributed", help="is distributed evaluation", default=1, type=int)
    parser.add_argument("--backend", help="", default='nccl', type=str)
    parser.add_argument("--dist_url", help="", default='127.0.0.1:41111', type=str)
    parser.add_argument("--gpus", help="number of gpus per node", default=1, type=int)
    parser.add_argument("--dist_rank", help="node rank for distributed training", default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = prepare_parser()

    # distributed or only one device
    if args.device_type == 'gpu':
        device = torch.device(f"cuda:{args.device_id}")
    elif args.device_type == 'npu':
        device = torch.device(f"npu:{args.device_id}")
        torch.npu.set_device(device)
    else:
        raise ValueError('device type error,please choice in ["gpu","npu"]')

    args.is_master_node = not args.distributed or args.device_id == 0

    # distributed config
    addr, port = args.dist_url.split(':')
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    if 'RANK_SIZE' in os.environ:
        args.rank_size = int(os.environ['RANK_SIZE'])
        args.rank = args.dist_rank * args.rank_size + args.device_id
        args.world_size = args.gpus * args.rank_size
        args.batch_size = int(args.batch_size / args.rank_size)
    else:
        raise RuntimeError("init_distributed_mode failed.")

    torch.distributed.init_process_group(backend=args.backend, init_method="env://",
                                         world_size=args.world_size, rank=args.rank)

    # 构建模型
    model = build_model(args, device)

    # 构建数据集
    dataloader, embeddings, issame_list = build_data_loader(args)

    # start evaluation
    evaluation(args, model, dataloader, embeddings, issame_list, device)

