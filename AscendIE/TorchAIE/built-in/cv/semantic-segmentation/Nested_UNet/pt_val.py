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

import os
import yaml
import json
import argparse
import numpy as np
import torch
import torch_aie
from torch_aie import _enums
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

from ais_bench.infer.interface import InferSession

# from utils.datasets import create_dataloader
# from common.util.dataset import BatchDataLoader, evaluate
from model_pt import forward_nms_script

# def forward_nms_script(model, dataloader, batch_size, device_id):
#     pass

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
#                       rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
#     if rect and shuffle:
#         LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
#         shuffle = False
#     with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
#         dataset = LoadImagesAndLabels(path, imgsz, batch_size,
#                                       augment=augment,  # augmentation
#                                       hyp=hyp,  # hyperparameters
#                                       rect=rect,  # rectangular batches
#                                       cache_images=cache,
#                                       single_cls=single_cls,
#                                       stride=int(stride),
#                                       pad=pad,
#                                       image_weights=image_weights,
#                                       prefix=prefix)

#     batch_size = min(batch_size, len(dataset))
#     nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
#     sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
#     loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
#     return loader(dataset,
#                   batch_size=batch_size,
#                   shuffle=shuffle and sampler is None,
#                   num_workers=nw,
#                   sampler=sampler,
#                   pin_memory=True,
#                   collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset

def collate_fn(batch):
    # img, label, path, shapes = zip(*batch)  # transposed
    img = batch  # transposed
    # print(list(bytes(img[0])))
    # for i, l in enumerate(label):
    #     l[:, 0] = i  # add target image index for build_targets()
    # return torch.stack(img, 0)
    return img


def create_dataloader(data_path, batch_size, workers=8):
    dataset = []
    data_file_list = os.listdir(data_path)
    for fname in data_file_list:
        with open(os.path.join(data_path, fname), "rb") as f:
            res = f.read()
            dataset.append(res)

    batch_size = min(batch_size, len(dataset))
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  num_workers=nw,
                  sampler=None,
                  pin_memory=True,
                  collate_fn=collate_fn), dataset


def main(opt):
    # load model
    model = torch.jit.load(opt.model)
    torch_aie.set_device(opt.device_id)
    if opt.need_compile:
        inputs = []
        inputs.append(torch_aie.Input((opt.batch_size, 3, opt.img_size, opt.img_size)))
        model = torch_aie.compile(
            model,
            inputs=inputs,
            precision_policy=_enums.PrecisionPolicy.FP16,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=opt.soc_version,
            optimization_level=0)

    # load dataset
    # single_cls = False if opt.tag == '9.6.0' else opt
    # dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size, max(cfg["stride"]), single_cls=False, pad=0.5)[0]
    dataloader = create_dataloader(opt.data_path, opt.batch_size)[0]
    # inference & nms
    pred_results = forward_nms_script(model, dataloader, opt.batch_size, opt.device_id)
    data_file_list = os.listdir(opt.data_path)
    for index, input_fname in enumerate(data_file_list):
        result_fname = input_fname
        # print(pred_results[index])
        # print(pred_results[index].tolist())
        np.array(pred_results[index].numpy().tofile(os.path.join("result/res_tmp", result_fname)))
        # print(bytes(pred_results[index].tolist()))
        # bytes(pred_results[index])

    # pred_json_file = f"{opt.model.split('.')[0]}_{opt.tag}_predictions.json"

    # print(f'saving results to {pred_json_file}')
    # with open(pred_json_file, 'w') as f:
    #     json.dump(pred_results, f)

    # evaluate mAP
    # evaluate(opt.ground_truth_json, pred_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv3 offline model inference.')
    parser.add_argument('--data_path', type=str, default="prep_data", help='root dir for val images and annotations')
    parser.add_argument('--ground_truth_json', type=str, default="coco/instances_val2017.json",
                        help='annotation file path')
    parser.add_argument('--tag', type=str, default='9.6.0', help='yolov3 tags')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--model', type=str, default="nested_unet_torch_aie.pt", help='ts model path')
    parser.add_argument('--need_compile', action="store_true", help='if the loaded model needs to be compiled or not')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    opt = parser.parse_args()

    # with open(opt.cfg_file) as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # main(opt, cfg)
    main(opt)
